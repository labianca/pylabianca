# coding=utf-8

"""
The MIT License (MIT)

Copyright (c) 2015 Aleksander Alafuzoff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import division

import os
import warnings
import numpy as np
import datetime

HEADER_LENGTH = 16 * 1024  # 16 kilobytes of header
NCS_SAMPLES_PER_RECORD = 512

VOLT_SCALING = (1, u'V')
MILLIVOLT_SCALING = (1_000, u'mV')
MICROVOLT_SCALING = (1_000_000, u'µV')

"""
NCS_RECORD
----------
TimeStamp       - Cheetah/ATLAS timestamp for this record. This corresponds to
                  the sample time for the first data point in the Samples
                  array. This value is in microseconds.
ChannelNumber   - The channel number for this record. This is NOT the A/D
                  channel number
SampleFreq      - The sampling frequency (Hz) for the data stored in the
                  Samples Field in this record
NumValidSamples - Number of values in Samples containing valid data
Samples         - Data points for this record. Cheetah/ATLAS currently supports
                  512 data points per record. At this time, the Samples array
                  is a [512] array.
"""
NCS_RECORD = np.dtype(
    [('TimeStamp', np.uint64), ('ChannelNumber', np.uint32),
     ('SampleFreq', np.uint32), ('NumValidSamples', np.uint32),
     ('Samples', np.int16, NCS_SAMPLES_PER_RECORD)]
)

"""
NEV_RECORD
----------
stx           - Reserved
pkt_id        - ID for the originating system of this packet
pkt_data_size - This value should always be two (2)
TimeStamp     - Cheetah/ATLAS timestamp for this record. This value is in
                microseconds.
event_id      - ID value for this event
ttl           - Decimal TTL value read from the TTL input port
crc           - Record CRC check from Cheetah/ATLAS. Not used in consumer
                applications.
dummy1        - Reserved
dummy2        - Reserved
Extra         - Extra bit values for this event. This array has a fixed
                length of eight (8)
EventString   - Event string associated with this event record. This string
                consists of 127 characters plus the required null termination
                character. If the string is less than 127 characters, the
                remainder of the characters will be null.
"""
NEV_RECORD = np.dtype(
    [('stx', np.int16), ('pkt_id', np.int16), ('pkt_data_size', np.int16),
     ('TimeStamp', np.uint64), ('event_id', np.int16), ('ttl', np.int16),
     ('crc', np.int16), ('dummy1', np.int16), ('dummy2', np.int16),
     ('Extra', np.int32, 8), ('EventString', 'S', 128)]
)


def read_header(file_path):
    '''Reads and parses the header of a Neuralynx file.

    Parameters
    ----------
    filename : str
        Path to the file.

    Returns
    -------
    header : dict
        Dictionary representing the header.
    '''
    file_path = os.path.abspath(file_path)
    with open(file_path, 'rb') as fid:
        raw_header = read_raw_header(fid)

    header = parse_header(raw_header)
    return header


def read_raw_header(fid):
    # Read the raw header data (16 kb) from the file object fid. Restores the
    # position in the file object after reading.
    pos = fid.tell()
    fid.seek(0)
    raw_hdr = fid.read(HEADER_LENGTH).strip(b'\0')
    fid.seek(pos)

    return raw_hdr


def _get_field_value(hdr_lines, field_name):
    line_idx = [idx for idx, txt in enumerate(hdr_lines)
                if txt.startswith(field_name)]

    if len(line_idx) == 0:
        warnings.warn(f'Could not find {field_name} in Neuralynx header.')
        idx = None
        value = None
    else:
        idx = line_idx[0]
        value = ' '.join(hdr_lines[idx].split(' ')[1:])

    return idx, value


def parse_header(raw_hdr):
    # Parse the header string into a dictionary of name value pairs
    hdr = dict()

    # Decode the header as iso-8859-1 (the spec says ASCII, but there is at
    # least one case of 0xB5 in some headers)
    raw_hdr = raw_hdr.decode('iso-8859-1')

    # Neuralynx headers seem to start with a line identifying the file, so
    # let's check for it
    hdr_lines = [line.strip() for line in raw_hdr.split('\r\n') if line != '']
    if hdr_lines[0] != '######## Neuralynx Data File Header':
        warnings.warn('Unexpected start to header: ' + hdr_lines[0])

    # Try to read the original file path
    try:
        try:
            assert hdr_lines[1].split()[1:3] == ['File', 'Name']
            hdr[u'FileName']  = ' '.join(hdr_lines[1].split()[3:])
            new_way = False
            # hdr['save_path'] = hdr['FileName']
        except AssertionError:
            field_name = '-OriginalFileName'
            _, value = _get_field_value(hdr_lines, field_name)
            hdr[u'FileName'] = value
            new_way = True
    except:
        warnings.warn(
            'Unable to parse original file path from Neuralynx header: '
            + hdr_lines[1])
        new_way = True

    # Process lines with file opening and closing times
    if new_way:
        parse_rest_from = 1
        time_fields = list()
        ix, hdr[u'TimeCreated'] = _get_field_value(hdr_lines, '-TimeCreated')
        time_fields.append(ix)
        hdr[u'TimeOpened_dt'] = parse_neuralynx_time_string_new(
            hdr[u'TimeCreated'])

        ix, hdr[u'TimeClosed'] = _get_field_value(hdr_lines, '-TimeClosed')
        time_fields.append(ix)
        hdr[u'TimeClosed_dt'] = parse_neuralynx_time_string_new(
            hdr[u'TimeClosed'])
    else:
        parse_rest_from = 4
        hdr[u'TimeOpened'] = hdr_lines[2][3:]
        hdr[u'TimeOpened_dt'] = parse_neuralynx_time_string(hdr_lines[2])
        hdr[u'TimeClosed'] = hdr_lines[3][3:]
        hdr[u'TimeClosed_dt'] = parse_neuralynx_time_string(hdr_lines[3])


    # Read the parameters, assuming "-PARAM_NAME PARAM_VALUE" format
    for line_idx, line in enumerate(hdr_lines[parse_rest_from:]):
        try:
            # Ignore the dash and split PARAM_NAME and PARAM_VALUE
            parts = line[1:].split()
            name = parts[0]
            value = ' '.join(parts[1:])
            if not new_way or (line_idx + parse_rest_from) not in time_fields:
                hdr[name] = value
        except:
            if not new_way or (line_idx + parse_rest_from) not in time_fields:
                warnings.warn(
                    'Unable to parse parameter line from Neuralynx header: '
                    + line)

    return hdr


def read_records(fid, record_dtype, record_skip=0, count=None):
    # Read count records (default all) from the file object fid skipping the
    # first record_skip records. Restores the position of the file object
    # after reading.
    if count is None:
        count = -1

    pos = fid.tell()
    fid.seek(HEADER_LENGTH, 0)
    fid.seek(record_skip * record_dtype.itemsize, 1)
    rec = np.fromfile(fid, record_dtype, count=count)
    fid.seek(pos)

    return rec


def estimate_record_count(file_path, record_dtype):
    # Estimate the number of records from the file size
    file_size = os.path.getsize(file_path)
    file_size -= HEADER_LENGTH

    if file_size % record_dtype.itemsize != 0:
        warnings.warn(
            'File size is not divisible by record size (some bytes '
            'unaccounted for)')

    return file_size / record_dtype.itemsize


def parse_neuralynx_time_string(time_string):
    # Parse a datetime object from the idiosyncratic time string in Neuralynx
    # file headers
    try:
        tmp_date = [int(x) for x in time_string.split()[4].split('/')]
        str_split = time_string.split()[-1].replace('.', ':').split(':')
        tmp_time = [int(x) for x in str_split]
        tmp_microsecond = tmp_time[3] * 1000
    except:
        warnings.warn('Unable to parse time string from Neuralynx header: '
                      + time_string)
        return None
    else:
        return datetime.datetime(
            tmp_date[2], tmp_date[0], tmp_date[1],  # Year, month, day
            tmp_time[0], tmp_time[1], tmp_time[2],  # Hour, minute, second
            tmp_microsecond
        )


def parse_neuralynx_time_string_new(time_string):
    # Parse a datetime object from the idiosyncratic time string in Neuralynx
    # file headers
    try:
        if time_string == 'File was not closed properly':
            return None
        else:
            tmp_date = [int(x) for x in time_string.split()[0].split('/')]
            tmp_time = [int(x) for x in time_string.split()[-1].split(':')]
            date_list = tmp_date + tmp_time
            return datetime.datetime(*date_list)
    except:
        warnings.warn(
            'Unable to parse time string from Neuralynx header: '
            + time_string)
        return None


def check_ncs_records(records):
    # Check that all the records in the array are "similar"
    # (have the same sampling frequency etc.)
    
    # first check if empty - if so, skip other checks
    is_empty = len(records) == 0
    if is_empty:
        warnings.warn('The file does not contain any data to read (apart '
                      'from the header')
        return False
    
    dt = np.diff(records['TimeStamp']).astype(int)  # uint by default
    dt = np.abs(dt - dt[0])
    good_n_valid_samples = records['NumValidSamples'] == 512

    if not np.all(records['ChannelNumber'] == records[0]['ChannelNumber']):
        warnings.warn('Channel number changed during record sequence')
        return False
    elif not np.all(records['SampleFreq'] == records[0]['SampleFreq']):
        warnings.warn('Sampling frequency changed during the sequence '
                      'of records')
        return False
    elif not np.all(good_n_valid_samples[:-1]):
        # [:-1] above is to ignore the last record - can be incomplete
        n_bad = np.sum(~good_n_valid_samples)
        warnings.warn(f'Invalid samples in {n_bad} records')
        return False
    elif not np.all(dt <= 1):
        warnings.warn('Time stamp difference tolerance exceeded')
        return False
    else:
        return True


def load_ncs(file_path, load_time=True, rescale_data=True,
             signal_scaling=MICROVOLT_SCALING):
    # Load the given file as a Neuralynx .ncs continuous acquisition file and
    # extract the contents
    file_path = os.path.abspath(file_path)
    with open(file_path, 'rb') as fid:
        raw_header = read_raw_header(fid)
        records = read_records(fid, NCS_RECORD)

    header = parse_header(raw_header)
    check_ncs_records(records)

    # Reshape (and rescale, if requested) the data into a 1D array
    data = records['Samples'].ravel()
    # data = records['Samples'].reshape(
    #     (NCS_SAMPLES_PER_RECORD * len(records), 1))
    if rescale_data:
        try:
            # ADBitVolts specifies the conversion factor between the ADC
            # counts and volts
            data = data.astype(np.float64) * (np.float64(header['ADBitVolts'])
                                              * signal_scaling[0])
        except KeyError:
            warnings.warn('Unable to rescale data, ADBitVolts value '
                          'not specified in the header')
            rescale_data = False

    # Pack the extracted data in a dictionary that is passed out of the function
    ncs = dict()
    ncs['file_path'] = file_path
    ncs['raw_header'] = raw_header
    ncs['header'] = header
    ncs['data'] = data
    ncs['data_units'] = signal_scaling[1] if rescale_data else 'ADC counts'
    
    n_records = len(records)
    if n_records > 0:
        ncs['sampling_rate'] = records['SampleFreq'][0]
        ncs['channel_number'] = records['ChannelNumber'][0]

    ncs['timestamp'] = records['TimeStamp']

    # Calculate the sample time points (if needed)
    if load_time:
        if n_records > 0:
            num_samples = data.shape[0]
            times = np.interp(
                np.arange(num_samples), np.arange(0, num_samples, 512),
                records['TimeStamp']).astype(np.uint64)
            ncs['time'] = times
        else:
            ncs['time'] = np.array([], dtype=np.uint64)
        ncs['time_units'] = u'µs'

    return ncs


def load_nev(file_path):
    # Load the given file as a Neuralynx .nev event file and extract the
    # contents
    file_path = os.path.abspath(file_path)
    with open(file_path, 'rb') as fid:
        raw_header = read_raw_header(fid)
        records = read_records(fid, NEV_RECORD)

    header = parse_header(raw_header)

    # Check for the packet data size, which should be two. DISABLED because
    # these seem to be set to 0 in our files.
    # assert np.all(record['pkt_data_size'] == 2), 'Some packets have invalid data size'

    # Pack the extracted data in a dictionary that is passed out of the
    # function
    nev = dict(file_path=file_path, raw_header=raw_header,
               header=header, records=records)
    nev['events'] = records[['pkt_id', 'TimeStamp', 'event_id', 'ttl',
                             'Extra', 'EventString']]

    return nev