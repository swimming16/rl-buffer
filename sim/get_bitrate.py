# Copyright (c) 2018, Kevin Spiteri
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import json
import math
import sys
import string
import os
import imp
from collections import namedtuple
from enum import Enum
import env
import ABR
VERBOSE=False
# Units used throughout:
#     size     : bits
#     time     : ms
#     size/time: bits/ms = kbit/s


# global variables:
#     video manifest
#     buffer contents
#     buffer first segment consumed
#     throughput estimate
#     latency estimate
#     rebuffer event count
#     rebuffer total time
#     session info


def load_json(path):
    with open(path) as file:
        obj = json.load(file)
    return obj

ManifestInfo = namedtuple('ManifestInfo', 'segment_time bitrates utilities segments')
NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency')

DownloadProgress = namedtuple('DownloadProgress',
                              'index quality '
                              'size downloaded '
                              'time time_to_first_bit '
                              'abandon_to_quality')

def get_buffer_level():
    global manifest
    global buffer_contents
    global buffer_fcc

    return manifest.segment_time * len(buffer_contents) - buffer_fcc

def deplete_buffer(time):
    global manifest
    global buffer_contents
    global buffer_fcc
    global rebuffer_event_count
    global rebuffer_time
    global played_utility
    global played_bitrate
    global total_play_time
    global total_bitrate_change
    global total_log_bitrate_change
    global last_played

    global rampup_origin
    global rampup_time
    global rampup_threshold
    global sustainable_quality

    if len(buffer_contents) == 0:
        rebuffer_time += time
        total_play_time += time
        return

    if buffer_fcc > 0:
        # first play any partial chunk left

        if time + buffer_fcc < manifest.segment_time:
            buffer_fcc += time
            total_play_time += time
            return

        time -= manifest.segment_time - buffer_fcc
        total_play_time += manifest.segment_time - buffer_fcc
        buffer_contents.pop(0)
        buffer_fcc = 0

    # buffer_fcc == 0 if we're here

    while time > 0 and len(buffer_contents) > 0:

        quality = buffer_contents[0]
        played_utility += manifest.utilities[quality]
        played_bitrate += manifest.bitrates[quality]
        if quality != last_played and last_played != None:
            total_bitrate_change += abs(manifest.bitrates[quality] -
                                        manifest.bitrates[last_played])
            total_log_bitrate_change += abs(math.log(manifest.bitrates[quality] /
                                                     manifest.bitrates[last_played]))
        last_played = quality

        if rampup_time == None:
            rt = sustainable_quality if rampup_threshold == None else rampup_threshold
            if quality >= rt:
                rampup_time = total_play_time - rampup_origin

        # bookkeeping to track reaction time to increased bandwidth
        for p in pending_quality_up:
            if len(p) == 2 and quality >= p[1]:
                p.append(total_play_time)

        if time >= manifest.segment_time:
            buffer_contents.pop(0)
            total_play_time += manifest.segment_time
            time -= manifest.segment_time
        else:
            buffer_fcc = time
            total_play_time += time
            time = 0

    if time > 0:
        rebuffer_time += time
        total_play_time += time
        rebuffer_event_count += 1

    process_quality_up(total_play_time)

def playout_buffer():
    global buffer_contents
    global buffer_fcc

    deplete_buffer(get_buffer_level())

    # make sure no rounding error
    del buffer_contents[:]
    buffer_fcc = 0

def process_quality_up(now):
    global max_buffer_size
    global pending_quality_up
    global total_reaction_time

    # check which switches can be processed

    cutoff = now - max_buffer_size
    while len(pending_quality_up) > 0 and pending_quality_up[0][0] < cutoff:
        p = pending_quality_up.pop(0)
        if len(p) == 2:
            reaction = max_buffer_size
        else:
            reaction = min(max_buffer_size, p[2] - p[0])
        #print('\n[%d] reaction time: %d' % (now, reaction))
        #print(p)
        total_reaction_time += reaction

def advertize_new_network_quality(quality, previous_quality):
    global max_buffer_size
    global network_total_time
    global pending_quality_up
    global buffer_contents

    # bookkeeping to track reaction time to increased bandwidth

    # process any previous quality up switches that have "matured"
    process_quality_up(network_total_time)

    # mark any pending switch up done if new quality switches back below its quality
    for p in pending_quality_up:
        if len(p) == 2 and p[1] > quality:
            p.append(network_total_time)
    #pending_quality_up = [p for p in pending_quality_up if p[1] >= quality]

    # filter out switches which are not upwards (three separate checks)
    if quality <= previous_quality:
        return
    for q in buffer_contents:
        if quality <= q:
            return
    for p in pending_quality_up:
        if quality <= p[1]:
            return

    # valid quality up switch
    pending_quality_up.append([network_total_time, quality])

class NetworkModel:

    min_progress_size = 12000
    min_progress_time = 50

    def __init__(self, network_trace):
        global sustainable_quality
        global network_total_time

        sustainable_quality = None
        network_total_time = 0
        self.trace = network_trace
        self.index = -1
        self.time_to_next = 0
        self.next_network_period()

    def next_network_period(self):
        global manifest
        global sustainable_quality
        global network_total_time

        self.index += 1
        if self.index == len(self.trace):
            self.index = 0
        self.time_to_next = self.trace[self.index].time

        latency_factor = 1 - self.trace[self.index].latency / manifest.segment_time
        effective_bandwidth = self.trace[self.index].bandwidth * latency_factor

        previous_sustainable_quality = sustainable_quality
        sustainable_quality = 0
        for i in range(1, len(ABR.VIDEO_BIT_RATE)):
            if ABR.VIDEO_BIT_RATE[i] > effective_bandwidth:
                break
            sustainable_quality = i
        if (sustainable_quality != previous_sustainable_quality and
            previous_sustainable_quality != None):
            advertize_new_network_quality(sustainable_quality, previous_sustainable_quality)

        if verbose:
            print('[%d] Network: %d,%d  (q=%d: bitrate=%d)' %
                  (round(network_total_time),
                   self.trace[self.index].bandwidth, self.trace[self.index].latency,
                   sustainable_quality, ABR.VIDEO_BIT_RATE[sustainable_quality]))

    # return delay time
    def do_latency_delay(self, delay_units):
        global network_total_time

        total_delay = 0
        while delay_units > 0:
            current_latency = self.trace[self.index].latency
            time = delay_units * current_latency
            if time <= self.time_to_next:
                total_delay += time
                network_total_time += time
                self.time_to_next -= time
                delay_units = 0
            else:
                # time > self.time_to_next implies current_latency > 0
                total_delay += self.time_to_next
                network_total_time += self.time_to_next
                delay_units -= self.time_to_next / current_latency
                self.next_network_period()
        return total_delay

    # return download time
    def do_download(self, size):
        global network_total_time
        total_download_time = 0
        while size > 0:
            current_bandwidth = self.trace[self.index].bandwidth
            if size <= self.time_to_next * current_bandwidth:
                # current_bandwidth > 0
                time = size / current_bandwidth
                total_download_time += time
                network_total_time += time
                self.time_to_next -= time
                size = 0
            else:
                total_download_time += self.time_to_next
                network_total_time += self.time_to_next
                size -= self.time_to_next * current_bandwidth
                self.next_network_period()
        return total_download_time

    def do_minimal_latency_delay(self, delay_units, min_time):
        global network_total_time
        total_delay_units = 0
        total_delay_time = 0
        while delay_units > 0 and min_time > 0:
            current_latency = self.trace[self.index].latency
            time = delay_units * current_latency
            if time <= min_time and time <= self.time_to_next:
                units = delay_units
                self.time_to_next -= time
                network_total_time += time
            elif min_time <= self.time_to_next:
                # time > 0 implies current_latency > 0
                time = min_time
                units = time / current_latency
                self.time_to_next -= time
                network_total_time += time
            else:
                time = self.time_to_next
                units = time / current_latency
                network_total_time += time
                self.next_network_period()
            total_delay_units += units
            total_delay_time += time
            delay_units -= units
            min_time -= time
        return (total_delay_units, total_delay_time)

    def do_minimal_download(self, size, min_size, min_time):
        global network_total_time
        total_size = 0
        total_time = 0
        while size > 0 and (min_size > 0 or min_time > 0):
            current_bandwidth = self.trace[self.index].bandwidth
            if current_bandwidth > 0:
                min_bits = max(min_size, min_time * current_bandwidth)
                bits_to_next = self.time_to_next * current_bandwidth
                if size <= min_bits and size <= bits_to_next:
                    bits = size
                    time = bits / current_bandwidth
                    self.time_to_next -= time
                    network_total_time += time
                elif min_bits <= bits_to_next:
                    bits = min_bits
                    time = bits / current_bandwidth
                    # make sure rounding error does not push while loop into endless loop
                    min_size = 0
                    min_time = 0
                    self.time_to_next -= time
                    network_total_time += time
                else:
                    bits = bits_to_next
                    time = self.time_to_next
                    network_total_time += time
                    self.next_network_period()
            else: # current_bandwidth == 0
                bits = 0
                if min_size > 0 or min_time > self.time_to_next:
                    time = self.time_to_next
                    network_total_time += time
                    self.next_network_period()
                else:
                    time = min_time
                    self.time_to_next -= time
                    network_total_time += time
            total_size += bits
            total_time += time
            size -= bits
            min_size -= bits
            min_time -= time
        return (total_size, total_time)

    def delay(self, time):
        global network_total_time
        while time > self.time_to_next:
            time -= self.time_to_next
            network_total_time += self.time_to_next
            self.next_network_period()
        self.time_to_next -= time
        network_total_time += time

    def download(self, size, idx, quality, buffer_level, check_abandon = None):
        if size <= 0:
            return DownloadProgress(index = idx, quality = quality,
                                    size = 0, downloaded = 0,
                                    time = 0, time_to_first_bit = 0,
                                    abandon_to_quality = None)

        if not check_abandon or (NetworkModel.min_progress_time <= 0 and
                                 NetworkModel.min_progress_size <= 0):
            latency = self.do_latency_delay(1)
            time = latency + self.do_download(size)
            return DownloadProgress(index = idx, quality = quality,
                                    size = size, downloaded = size,
                                    time = time, time_to_first_bit = latency,
                                    abandon_to_quality = None)

        total_download_time = 0
        total_download_size = 0
        min_time_to_progress = NetworkModel.min_progress_time
        min_size_to_progress = NetworkModel.min_progress_size

        if NetworkModel.min_progress_size > 0:
            latency = self.do_latency_delay(1)
            total_download_time += latency
            min_time_to_progress -= total_download_time
            delay_units = 0
        else:
            latency = None
            delay_units = 1

        abandon_quality = None
        while total_download_size < size and abandon_quality == None:

            if delay_units > 0:
                # NetworkModel.min_progress_size <= 0
                (units, time) = self.do_minimal_latency_delay(delay_units, min_time_to_progress)
                total_download_time += time
                delay_units -= units
                min_time_to_progress -= time
                if delay_units <= 0:
                    latency = total_download_time

            if delay_units <= 0:
                # don't use else to allow fall through
                (bits, time) = self.do_minimal_download(size - total_download_size,
                                                        min_size_to_progress, min_time_to_progress)
                total_download_time += time
                total_download_size += bits
                # no need to upldate min_[time|size]_to_progress - reset below

            dp = DownloadProgress(index = idx, quality = quality,
                                  size = size, downloaded = total_download_size,
                                  time = total_download_time, time_to_first_bit = latency,
                                  abandon_to_quality = None)
            if total_download_size < size:
                abandon_quality = check_abandon(dp, max(0, buffer_level - total_download_time))
                if abandon_quality != None:
                    if verbose:
                        print('%d abandoning %d->%d' % (idx, quality, abandon_quality))
                        print('%d/%d %d(%d)' %
                              (dp.downloaded, dp.size, dp.time, dp.time_to_first_bit))
                min_time_to_progress = NetworkModel.min_progress_time
                min_size_to_progress = NetworkModel.min_progress_size

        return DownloadProgress(index = idx, quality = quality,
                                size = size, downloaded = total_download_size,
                                time = total_download_time, time_to_first_bit = latency,
                                abandon_to_quality = abandon_quality)

class ThroughputHistory:
    def __init__(self, config):
        pass
    def push(self, time, tput, lat):
        raise NotImplementedError

class SessionInfo:

    def __init__(self):
        pass

    def get_throughput(self):
        global throughput
        return throughput

    def get_buffer_contents(self):
        global buffer_contents
        return buffer_contents[:]

session_info = SessionInfo()

class Abr:

    session = session_info

    def __init__(self, config):
        pass
    def get_quality_delay(self, segment_index):
        raise NotImplementedError
    def get_first_quality(self):
        return 0
    def report_delay(self, delay):
        pass
    def report_download(self, metrics, is_replacment):
        pass
    def report_seek(self, where):
        pass
    def check_abandon(self, progress, buffer_level):
        return None

    def quality_from_throughput(self, tput,latency):

        p = int(env.VIDEO_CHUNCK_LEN)

        quality = 0
        while (quality + 1 < len(ABR.VIDEO_BIT_RATE) and
               latency + p * ABR.VIDEO_BIT_RATE[quality + 1] / tput <= p):
            quality += 1
        return quality

class Replacement:

    session = session_info

    def check_replace(self, quality):
        return None
    def check_abandon(self, progress, buffer_level):
        return None

average_list = {}
abr_list = {}

class SlidingWindow(ThroughputHistory):

    default_window_size = [3]
    max_store = 20

    def __init__(self, config):

        if 'window_size' in config and config['window_size'] != None:
            self.window_size = config['window_size']
        else:
            self.window_size = SlidingWindow.default_window_size

        # TODO: init somewhere else?
        throughput = None
        latency = None

        self.last_throughputs = []
        self.last_latencies = []

    def push(self, tput, lat):

        self.last_throughputs += [tput]
        self.last_throughputs = self.last_throughputs[-SlidingWindow.max_store:]

        self.last_latencies += [lat]
        self.last_latencies = self.last_latencies[-SlidingWindow.max_store:]

        tput = None
        lat = None
        for ws in self.window_size:
            sample = self.last_throughputs[-ws:]
            t = sum(sample) / len(sample)
            tput = t if tput == None else min(tput, t) # conservative min
            sample = self.last_latencies[-ws:]
            l = sum(sample) / len(sample)
            lat = l if lat == None else max(lat, l) # conservative max
        return tput

average_list['sliding'] = SlidingWindow

class Ewma(ThroughputHistory):

    # for throughput:
    default_half_life = [8000, 3000]

    def __init__(self, config):
        global throughput
        global latency

        # TODO: init somewhere else?
        throughput = None
        latency = None

        if 'half_life' in config and config['half_life'] != None:
            self.half_life = [h * 1000 for h in config['half_life']]
        else:
            self.half_life = Ewma.default_half_life

        self.latency_half_life = [h / manifest.segment_time for h in self.half_life]

        self.throughput = [0] * len(self.half_life)
        self.weight_throughput = 0
        self.latency = [0] * len(self.half_life)
        self.weight_latency = 0

    def push(self, time, tput, lat):
        global throughput
        global latency

        for i in range(len(self.half_life)):
            alpha = math.pow(0.5, time / self.half_life[i])
            self.throughput[i] = alpha * self.throughput[i] + (1 - alpha) * tput
            alpha = math.pow(0.5, 1 / self.latency_half_life[i])
            self.latency[i] = alpha * self.latency[i] + (1 - alpha) * lat

        self.weight_throughput += time
        self.weight_latency += 1

        tput = None
        lat = None
        for i in range(len(self.half_life)):
            zero_factor = 1 - math.pow(0.5, self.weight_throughput / self.half_life[i])
            t = self.throughput[i] / zero_factor
            tput = t if tput == None else min(tput, t)  # conservative case is min
            zero_factor = 1 - math.pow(0.5, self.weight_latency / self.latency_half_life[i])
            l = self.latency[i] / zero_factor
            lat = l if lat == None else max(lat, l) # conservative case is max
        throughput = tput
        latency = lat

average_list['ewma'] = Ewma
average_default = 'ewma'

class Bola(Abr):
    def __init__(self, config):
        global verbose

        utility_offset = -math.log(ABR.VIDEO_BIT_RATE[0]) # so utilities[0] = 0
        self.utilities = [math.log(b) + utility_offset for b in ABR.VIDEO_BIT_RATE]

        self.gp = config['gp']
        self.buffer_size = config['buffer_size']
        self.abr_osc = config['abr_osc']
        self.abr_basic = config['abr_basic']
        self.Vp = config['Vp']


        self.last_seek_index = 0 # TODO
        self.last_quality = 0

        if VERBOSE:
            for q in range(len(ABR.VIDEO_BIT_RATE)):
                b = ABR.VIDEO_BIT_RATE[q]
                u = self.utilities[q]
                l = self.Vp * (self.gp + u)
                if q == 0:
                    print('%d %d' % (q, l))
                else:
                    qq = q - 1
                    bb = ABR.VIDEO_BIT_RATE[qq]
                    uu = self.utilities[qq]
                    ll = self.Vp * (self.gp + (b * uu - bb * u) / (b - bb))
                    print('%d %d    <- %d %d' % (q, l, qq, ll))
    def compute_throughput(self,throughput,latency):
        sliding=SlidingWindow([3])
        throughput=sliding.push(throughput,latency)
        #print(latency)
        return throughput

    def quality_from_buffer(self,level):
        # print('l',level)
        quality = 0
        score = None
        for q in range(len(ABR.VIDEO_BIT_RATE)):
            s = ((self.Vp * (self.utilities[q] + self.gp) - level) / ABR.VIDEO_BIT_RATE[q])
            if score == None or s > score:
                quality = q
                score = s
        return quality

    def get_quality(self,Vp, buffer_level,last_quality,throughput):
        #global manifest
        #global throughput

        if not self.abr_basic:
            # t = min(segment_index - self.last_seek_index, len(manifest.segments) - segment_index)
            # t = max(t / 2, 3)
            # t = t * manifest.segment_time
            #buffer_size = min(self.buffer_size, t)
            # self.Vp = (buffer_size - manifest.segment_time) / (self.utilities[-1] + self.gp)
            self.Vp=Vp

        quality = self.quality_from_buffer(buffer_level)

        # if quality > last_quality:
        #     throughput=self.compute_throughput(throughput,env.LINK_RTT)
        #     quality_t = self.quality_from_throughput(throughput,env.LINK_RTT)
        #     if quality <= quality_t:
        #         delay = 0
        #     elif last_quality > quality_t:
        #         quality = last_quality
        #         delay = 0
        #     else:
        #         if not self.abr_osc:
        #             quality = quality_t + 1
        #             delay = 0
        #         else:
        #             quality = quality_t
        #             # now need to calculate delay
        #             b = ABR.VIDEO_BIT_RATE[quality]
        #             u = self.utilities[quality]
        #             #bb = ABR.VIDEO_BIT_RATE[quality + 1]
        #             #uu = self.utilities[quality + 1]
        #             #l = self.Vp * (self.gp + (bb * u - b * uu) / (bb - b))
        #             l = self.Vp * (self.gp + u) ##########
        #             delay = max(0, get_buffer_level() - l)
        #             if quality == len(ABR.VIDEO_BIT_RATE) - 1:
        #                 delay = 0
        #             # delay = 0 ###########

        return quality

    def report_seek(self, where):
        # TODO: seek properly
        global manifest
        self.last_seek_index = math.floor(where / manifest.segment_time)

    def check_abandon(self, progress, buffer_level):
        global manifest

        if self.abr_basic:
            return None

        remain = progress.size - progress.downloaded
        if progress.downloaded <= 0 or remain <= 0:
            return None

        abandon_to = None
        score = (self.Vp * (self.gp + self.utilities[progress.quality]) - buffer_level) / remain
        if score < 0:
            return # TODO: check

        for q in range(progress.quality):
            other_size = progress.size * ABR.VIDEO_BIT_RATE[q] / ABR.VIDEO_BIT_RATE[progress.quality]
            other_score = (self.Vp * (self.gp + self.utilities[q]) - buffer_level) / other_size
            if other_size < remain and other_score > score:
                # check size: see comment in BolaEnh.check_abandon()
                score = other_score
                abandon_to = q

        if abandon_to != None:
            self.last_quality = abandon_to

        return abandon_to

abr_list['bola'] = Bola



abr_default = 'bola'

#print(abr_list)

class NoReplace(Replacement):
        pass

# TODO: different classes instead of strategy
class Replace(Replacement):

    def __init__(self, strategy):
        self.strategy = strategy
        self.replacing = None
        # self.replacing is either None or -ve index to buffer_contents

    def check_replace(self, quality):
        global manifest
        global buffer_contents
        global buffer_fcc

        self.replacing = None

        if self.strategy == 0:

            skip = math.ceil(1.5 + buffer_fcc / manifest.segment_time)
            #print('skip = %d  fcc = %d' % (skip, buffer_fcc))
            for i in range(skip, len(buffer_contents)):
                if buffer_contents[i] < quality:
                    self.replacing = i - len(buffer_contents)
                    break

            #if self.replacing == None:
            #    print('no repl:  0/%d' % len(buffer_contents))
            #else:
            #    print('replace: %d/%d' % (self.replacing, len(buffer_contents)))

        elif self.strategy == 1:

            skip = math.ceil(1.5 + buffer_fcc / manifest.segment_time)
            #print('skip = %d  fcc = %d' % (skip, buffer_fcc))
            for i in range(len(buffer_contents) - 1, skip - 1, -1):
                if buffer_contents[i] < quality:
                    self.replacing = i - len(buffer_contents)
                    break

            #if self.replacing == None:
            #    print('no repl:  0/%d' % len(buffer_contents))
            #else:
            #    print('replace: %d/%d' % (self.replacing, len(buffer_contents)))

        else:
            pass


        return self.replacing

    def check_abandon(self, progress, buffer_level):
        global manifest
        global buffer_contents
        global buffer_fcc

        if self.replacing == None:
            return None
        if buffer_level + manifest.segment_time * self.replacing <= 0:
            return -1
        return None


class AbrInput(Abr):

    def __init__(self, path, config):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.abr_module = imp.load_source(self.name, path)
        self.abr_class = getattr(self.abr_module, self.name)
        self.abr_class.session = session_info
        self.abr = self.abr_class(config)

    def get_quality_delay(self, segment_index):
        return self.abr.get_quality_delay(segment_index)
    def get_first_quality(self):
        return self.abr.get_first_quality()
    def report_delay(self, delay):
        self.abr.report_delay(delay)
    def report_download(self, metrics, is_replacment):
        self.abr.report_download(metrics, is_replacment)
    def report_seek(self, where):
        self.abr.report_seek(where)
    def check_abandon(self, progress, buffer_level):
        return self.abr.check_abandon(progress, buffer_level)

class ReplacementInput(Replacement):

    def __init__(self, path):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.replacement_module = imp.load_source(self.name, path)
        self.replacement_class = getattr(self.replacement_module, self.name)
        self.replacement_class.session = session_info
        self.replacement = self.replacement_class()

    def check_replace(self, quality):
        return self.replacement.check_replace(quality)
    def check_abandon(self, progress, buffer_level):
        return self.replacement.check_abandon(progress, buffer_level)

