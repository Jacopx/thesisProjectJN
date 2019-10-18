from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from geopy import distance
import re
import sys
import time
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import requests
warnings.simplefilter('ignore')


subtypo = {'Potentially Life-Threatening': 1, 'Non Life-threatening': 2, 'Alarm': 3, 'Fire': 4}
subtypoR = {1: 'Potentially Life-Threatening', 2: 'Non Life-threatening', 3: 'Alarm', 4: 'Fire'}

typo = {'Medical Incident': 0, 'Outside Fire': 1, 'Alarms': 2, 'Citizen Assist / Service Call': 3,
        'Traffic Collision': 4, 'Other': 5, 'Structure Fire': 6, 'Smoke Investigation (Outside)': 7,
        'Electrical Hazard': 8, 'Elevator / Escalator Rescue': 9, 'Vehicle Fire': 10,
        'Gas Leak (Natural and LP Gases)': 11, 'Water Rescue': 12, 'Odor (Strange / Unknown)': 13, 'Fuel Spill': 14,
        'Train / Rail Incident': 15, 'Administrative': 16, 'Marine Fire': 17, 'Industrial Accidents': 18,
        'High Angle Rescue': 19, 'HazMat': 20, 'Explosion': 21, 'Confined Space / Structure Collapse': 22,
        'Assist Police': 23, 'Extrication / Entrapped (Machinery, Vehicle)': 24, 'Watercraft in Distress': 25,
        'Suspicious Package': 26, 'Train / Rail Fire': 27, 'Mutual Aid / Assist Outside Agency': 28,
        'Lightning Strike (Investigation)': 29, 'Aircraft Emergency': 30, 'Oil Spill': 31}
typoR = {0: 'Medical Incident', 1: 'Outside Fire', 2: 'Alarms', 3: 'Citizen Assist / Service Call',
         4: 'Traffic Collision', 5: 'Other', 6: 'Structure Fire', 7: 'Smoke Investigation (Outside)',
         8: 'Electrical Hazard', 9: 'Elevator / Escalator Rescue', 10: 'Vehicle Fire',
         11: 'Gas Leak (Natural and LP Gases)', 12: 'Water Rescue', 13: 'Odor (Strange / Unknown)', 14: 'Fuel Spill',
         15: 'Train / Rail Incident', 16: 'Administrative', 17: 'Marine Fire', 18: 'Industrial Accidents',
         19: 'High Angle Rescue', 20: 'HazMat', 21: 'Explosion', 22: 'Confined Space / Structure Collapse',
         23: 'Assist Police', 24: 'Extrication / Entrapped (Machinery, Vehicle)', 25: 'Watercraft in Distress',
         26: 'Suspicious Package', 27: 'Train / Rail Fire', 28: 'Mutual Aid / Assist Outside Agency',
         29: 'Lightning Strike (Investigation)', 30: 'Aircraft Emergency', 31: 'Oil Spill'}

unit = {'E23': 0, '83': 1, 'E35': 2, 'E10': 3, 'E03': 4, 'QRV1': 5, '68': 6, '71': 7, 'E14': 8, '75': 9, 'KM14': 10, 'E07': 11, '87': 12, '84': 13, 'E17': 14, '60': 15, '62': 16, 'KM09': 17, 'E36': 18, 'E19': 19, '79': 20, '70': 21, '82': 22, '86': 23, 'E15': 24, 'T15': 25, '66': 26, '81': 27, 'RC4': 28, 'AM118': 29, 'E38': 30, '85': 31, 'T05': 32, 'AM116': 33, 'T08': 34, 'E08': 35, '54': 36, 'B04': 37, 'E01': 38, 'T03': 39, 'RS1': 40, 'T01': 41, 'D2': 42, 'B01': 43, '59': 44, 'E05': 45, 'E41': 46, '94': 47, '76': 48, 'E16': 49, 'KM07': 50, 'T06': 51, 'B02': 52, 'E06': 53, 'E28': 54, 'RS2': 55, '78': 56, '93': 57, 'B09': 58, 'E18': 59, 'E40': 60, 'T02': 61, 'T16': 62, 'EMS6': 63, 'E37': 64, 'E33': 65, 'E11': 66, 'E32': 67, 'E22': 68, 'E44': 69, 'B10': 70, 'T17': 71, 'E31': 72, 'T19': 73, 'B08': 74, 'E48': 75, 'T48': 76, 'T13': 77, 'E13': 78, 'KM06': 79, '74': 80, 'T09': 81, 'E29': 82, 'RC1': 83, 'E39': 84, 'T12': 85, 'E20': 86, '91': 87, 'RC3': 88, 'T11': 89, 'B06': 90, 'E24': 91, '58': 92, 'E21': 93, 'AM114': 94, 'E12': 95, '53': 96, 'E09': 97, 'E25': 98, '50': 99, 'AM110': 100, 'AM150': 101, '72': 102, 'AM106': 103, 'B03': 104, '89': 105, 'E51': 106, 'KM11': 107, '52': 108, 'FB3': 109, '55': 110, 'AM291': 111, '63': 112, 'QRV2': 113, 'AM102': 114, 'AM104': 115, '65': 116, 'KM13': 117, 'KM02': 118, 'E02': 119, 'E42': 120, 'RC2': 121, 'E34': 122, 'T10': 123, 'B05': 124, 'D3': 125, 'T07': 126, 'E04': 127, 'AM112': 128, 'T18': 129, 'B07': 130, 'E43': 131, 'E26': 132, 'T04': 133, 'RA48': 134, 'T14': 135, 'AM120': 136, 'AM108': 137, '56': 138, '67': 139, '57': 140, 'SR34': 141, '64': 142, 'AM126': 143, 'HT48': 144, 'AM124': 145, '95': 146, 'KM12': 147, '77': 148, 'AR1': 149, '61': 150, 'KM08': 151, 'KM04': 152, '96': 153, 'AM128': 154, 'KM01': 155, 'AM122': 156, 'EMS6A': 157, 'AM130': 158, '73': 159, 'ALS920': 160, 'AM238': 161, 'AM235': 162, '99': 163, 'AM234': 164, 'AM229': 165, 'AM505': 166, '88': 167, 'AM237': 168, 'AM242': 169, 'CR19': 170, 'KM204': 171, 'MP44': 172, 'MP25': 173, 'RB1': 174, 'CR14': 175, 'SR18': 176, 'BT01': 177, 'EMS2': 178, 'MD1': 179, 'BPRI': 180, 'KM10': 181, 'ISB1': 182, 'MA1': 183, 'BE1': 184, 'SO1': 185, 'RWC1': 186, 'RWC2': 187, 'SO2': 188, 'AM245': 189, 'BLS800': 190, 'KM03': 191, 'FB1': 192, 'AM154': 193, 'AM152': 194, 'MP32': 195, 'SO4': 196, 'MP43': 197, 'AP': 198, 'AR2': 199, 'AM293': 200, 'AM156': 201, 'PAT1': 202, 'PAT2': 203, 'ROCK4': 204, 'ROCK1': 205, 'PTRL2': 206, 'ROCK3': 207, 'ROCK2': 208, 'GATOR3': 209, 'PTRL1': 210, 'ROCK7': 211, 'AM158': 212, 'GATOR2': 213, 'GATOR4': 214, 'ROCK8': 215, 'ROCK': 216, 'BLS822': 217, 'AM213': 218, 'AM215': 219, 'AM218': 220, 'AM219': 221, 'AM210': 222, 'ALS925': 223, 'ALS922': 224, 'ALS921': 225, 'ALS924': 226, 'ALS926': 227, 'ALS923': 228, 'BLS803': 229, 'PT107': 230, 'BLS820': 231, 'PT707': 232, 'BLS802': 233, 'BE2': 234, 'HZ1': 235, 'EMS6B': 236, 'CD1': 237, 'CD2': 238, 'GATOR1': 239, 'AM233': 240, 'AM222': 241, 'KM15': 242, 'S04': 243, 'BSEC': 244, 'EMS1': 245, 'BLS801': 246, 'AM132': 247, 'AM247': 248, 'KM14A': 249, 'PC1': 250, 'OES361': 251, 'E82': 252, 'E83': 253, 'E81': 254, 'E84': 255, 'E85': 256, 'AM239': 257, 'FST200': 258, 'FOOT1': 259, 'AM501': 260, '47': 261, '49': 262, 'SFR1': 263, 'PT204': 264, 'PT205': 265, 'PT103': 266, 'PT203': 267, 'MED1': 268, 'MED4': 269, 'MED2': 270, 'MED3': 271, 'PT100': 272, 'PT106': 273, 'MED5': 274, 'AM160': 275, 'E73': 276, 'E72': 277, 'E71': 278, 'E75': 279, 'E74': 280, 'CD3': 281, 'CO2': 282, 'AM200': 283, 'AM226': 284, 'KM09A': 285, 'FP203': 286, 'FP201': 287, 'KM01A': 288, 'QRV306': 289, 'AM164': 290, 'AM166': 291, 'EMS10': 292, 'AM50': 293, 'AM52': 294, 'SS1': 295, 'AM56': 296, 'AM305': 297, 'ALS900': 298, 'BLS805': 299, 'ALS901': 300, 'ALS902': 301, 'BLS804': 302, 'AM06': 303, 'AM02': 304, 'AM08': 305, 'AM18': 306, 'AM24': 307, 'AM10': 308, 'AM12': 309, 'AM26': 310, 'AM20': 311, 'AM22': 312, 'AM16': 313, 'AM14': 314, 'LR5': 315, 'AM04': 316, 'AM30': 317, 'AM28': 318, 'MCB2': 319, 'MCB1': 320, 'HSA1': 321, 'RC5': 322, 'SFR2': 323, 'AM54': 324, 'LR7': 325, 'AM36': 326, 'RC6': 327, 'TF2': 328, 'TF4': 329, 'AM304': 330, 'TF8': 331, 'TF5': 332, 'AM32': 333, 'TF6': 334, 'TF3': 335, 'AM227': 336, 'FA753': 337, 'AM97': 338, 'PP400': 339, 'AM99': 340, 'FA269': 341, 'AM98': 342, 'FA461': 343, 'SO50': 344, 'ST275': 345, 'SO18': 346, 'AM142': 347, 'SO19': 348, 'RL206': 349, 'AM201': 350, 'RL205': 351, 'AM208': 352, 'SJ114': 353, 'SM209': 354, 'SJ116': 355, 'PT200': 356, 'AM203': 357, 'AM217': 358, 'AM214': 359, 'AM209': 360, 'AM202': 361, 'ST212': 362, 'ST213': 363, 'AM225': 364, 'TI': 365, 'AM115': 366, 'PT102': 367, 'BLS2': 368, 'BLS1': 369, 'PT202': 370, 'AM129': 371, 'PT306': 372, '51': 373, 'AM64': 374, 'AM60': 375, 'CR1': 376, 'SR1': 377, 'GATOR5': 378, 'KM04A': 379, 'SH216': 380, 'SJ111': 381, 'BS105': 382, 'AM224': 383, 'AM137': 384, 'SH223': 385, 'MCT1': 386, 'MCU1': 387, 'KM05': 388, 'PT247': 389, 'AM05': 390, 'RB2': 391, 'PT9': 392, 'PT101': 393, 'PT201': 394, 'PT6': 395, 'PT2': 396, 'RM02': 397, 'RM05': 398, 'RM01': 399, 'AM58': 400, 'AM62': 401, '92': 402, 'PRO706': 403, 'PT270': 404, 'FAST2': 405, 'FAST3': 406, 'AM133': 407, 'FAST4': 408, 'FAST1': 409, 'GATOR6': 410, 'PT105': 411, 'FAST5': 412, 'AM135': 413, 'BS102': 414, 'BS01': 415, 'SJ113': 416, 'BS103': 417, 'BU1': 418, 'PT705': 419, 'PT332': 420, 'PT01': 421, 'GATR2': 422, 'GATR1': 423, 'AM193': 424, 'GATR3': 425, 'AM34': 426, 'BS205': 427, 'FHT48': 428, 'BS41': 429, 'PT06': 430, 'PT03': 431, 'AMR305': 432, 'BS107': 433, 'BS106': 434, 'STJ111': 435, 'AMR137': 436, 'PT05': 437, 'AM131': 438, 'AM02A': 439, 'ISB2': 440, 'PT708': 441, 'PT706': 442, 'BS206': 443, 'UU1': 444, 'S1': 445, 'MC1': 446, 'STJ110': 447, 'EMS4': 448, 'BA41': 449, 'AMRM24': 450, 'J1': 451, 'B84': 452, 'B82': 453, 'P255': 454, '90': 455, 'AMRM34': 456, 'AMRM20': 457, 'AMRM22': 458, 'AMRM26': 459, 'PT1': 460, 'PT7': 461, 'PT3': 462, 'PT5': 463, 'PT4': 464, 'PT10': 465, 'A304': 466, 'AM125': 467, 'RE864': 468, 'AMRM10': 469, 'KA205': 470, 'AMRM32': 471, 'AMRM28': 472, 'AMRM30': 473, 'KM02A': 474, 'AMRM6': 475, 'DCFD1': 476, 'SM21': 477, 'SM24': 478, 'AM07': 479, 'M51': 480, 'B11': 481, 'AM13': 482, 'OES248': 483, '27': 484, '80': 485, '30': 486, '45': 487, 'CC02': 488, 'M17': 489, 'M15': 490, 'AHT42': 491, 'M19': 492, 'DEME': 493, 'AMR6': 494, 'M22': 495, 'B15': 496, 'K1294': 497, 'A204': 498, 'K1307': 499, 'K1330': 500, 'K1264': 501, 'K1295': 502, 'A206': 503, 'A208': 504, 'K1217': 505, 'K1309': 506, 'K1313': 507, 'K1327': 508, 'K895': 509, 'K1302': 510, 'K847': 511, 'K1272': 512, 'K1265': 513, 'K1300': 514, 'A202': 515, 'A210': 516, 'SOC1': 517, 'AMR2': 518, 'AMR8': 519, 'A106': 520, 'FHT17': 521, 'K1268': 522, 'M43': 523, 'FB2': 524, 'K1306': 525, 'K1278': 526, 'M39': 527, 'M18': 528, 'M14': 529, 'M38': 530, 'KAAM14': 531, 'AMRM36': 532, 'KAAM2': 533, 'KAAM16': 534, 'KAAM18': 535, 'KAAM10': 536, 'AMREV': 537, 'AMRM12': 538, 'A104': 539, 'EMS7': 540, 'T51': 541, 'TF7': 542, 'K1301': 543, 'K1245': 544, 'K1210': 545, 'DCFD2': 546, 'K1270': 547, 'A102': 548, '98': 549, 'M21': 550, 'M11': 551, 'M28': 552, 'K1247': 553, 'M29': 554, 'M32': 555, 'K1271': 556, 'K917': 557, 'M13': 558, 'M09': 559, 'M12': 560, 'K1076': 561, 'K1209': 562, 'K1242': 563, 'M08': 564, 'M07': 565, 'M41': 566, 'A212': 567, 'MMTS': 568, 'A154': 569, 'K1105': 570, 'SO3': 571, 'K1244': 572, 'K1075': 573, 'K1207': 574, 'K1240': 575, 'M05': 576, 'M06': 577, 'K1169': 578, 'K1077': 579, 'M03': 580, 'M36': 581, 'M01': 582, 'M10': 583, 'K1181': 584, 'A108': 585, 'K996': 586, 'FHT22': 587, 'K1194': 588, 'A112': 589, 'A152': 590, 'M48': 591, 'K1190': 592, 'K968': 593, 'K1173': 594, 'RM719': 595, 'MA2': 596, 'A158': 597, 'RS01': 598, 'K439': 599, 'K1146': 600, 'E47': 601, 'MCU': 602, 'MP47': 603, 'K669': 604, 'K768': 605, 'K1098': 606, 'A214': 607, 'K1006': 608, 'K1092': 609, 'K735': 610, 'OES': 611, 'A216': 612, 'FAST26': 613, 'FAST40': 614, 'K1029': 615, 'MP9': 616, 'K922': 617, 'K834': 618, 'K720': 619, 'E53': 620, 'M57': 621, 'A156': 622, 'B50': 623, 'RC52': 624, 'D1': 625, 'RC51': 626, 'A63': 627, 'A62': 628, 'A61': 629, 'RC7': 630, 'K667': 631, 'FHT25': 632, 'K790': 633, 'AHT13': 634, 'K1005': 635, 'RA32': 636, 'RA13': 637, 'RA12': 638, 'RA29': 639, 'RA08': 640, 'RA10': 641, 'RA38': 642, 'RA09': 643, 'M52': 644, 'RA01': 645, 'M35': 646, 'K888': 647, 'RA35': 648, 'RA03': 649, 'RA41': 650, 'RA07': 651, 'RA05': 652, 'A917': 653, 'MP4': 654, 'K585': 655, 'M02': 656, 'MP1': 657, 'MP2': 658, 'A922': 659, 'M16': 660, 'RA33': 661, 'M33': 662, 'A931': 663, 'A933': 664, 'A802': 665, 'A782': 666, 'A837': 667, 'A929': 668, 'A786': 669, 'A921': 670, 'A897': 671, 'A727': 672, 'A889': 673}

stations = {'23': 0, '35': 1, '10': 2, '41': 3, '14': 4, '07': 5, '01': 6, '36': 7, '17': 8, '02': 9, '19': 10, '03': 11, '15': 12, '38': 13, '33': 14, '31': 15, '11': 16, '16': 17, '06': 18, '28': 19, '18': 20, '40': 21, '08': 22, '39': 23, '37': 24, '05': 25, '13': 26, '22': 27, '32': 28, '21': 29, '44': 30, '48': 31, '29': 32, '20': 33, '24': 34, '12': 35, '25': 36, '09': 37, '51': 38, '42': 39, '34': 40, '04': 41, '43': 42, '26': 43, 'nan': 44, 'E2': 45, '47': 46, '94': 47, 'F3': 48, 'A1': 49, 'A3': 50, 'A2': 51, '53': 52, '27': 53}


def dot():
    print('.', end='')


def replace_camelcase_with_underscore(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)


def clean_col_names(df):
    df.columns = [name.lstrip() for name in df.columns.tolist()]
    df.columns = [name.rstrip() for name in df.columns.tolist()]
    df.columns = [name.replace(' ', '') for name in df.columns.tolist()]
    df.columns = [name.replace('_', '') for name in df.columns.tolist()]
    df.columns = [replace_camelcase_with_underscore(name) for name in df.columns.tolist()]
    df.columns = [name.lower() for name in df.columns.tolist()]
    return df


def read_data(dest):
    print('Reading data...', end='')
    path = 'data'  # use your path
    all_files = glob.glob(path + "/" + dest + ".csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, low_memory=False, parse_dates=True, error_bad_lines=False)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    print(' OK')
    return frame


def data_reduction(df):
    print('Remove columns...', end='')
    df = df[['call_number', 'unit_id', 'unit_type', 'call_type', 'call_type_group', 'received_dt_tm', 'on_scene_dt_tm', 'available_dt_tm',
             'zipcodeof_incident', 'numberof_alarms', 'battalion', 'station_area', 'box', 'priority', 'location']]
    print(' OK')
    return df


def data_reduction2(df):
    print('Remove columns...', end='')
    df = df[['call_number', 'unit_id', 'unit_type', 'call_type', 'call_type_group', 'priority', 'numberof_alarms', 'rec_dt', 'onscene_dt', 'end_dt',
             'duration', 'res_time', 'rec_day', 'rec_month', 'rec_hour', 'rec_day_of_week', 'week', 'year', 'end_day', 'end_month', 'end_hour', 'end_day_of_week',
            'battalion', 'station_area', 'station_size', 'station_lat', 'station_long', 'zipcodeof_incident', 'box', 'lat', 'long']]
    print(' OK')
    return df


def location(df):
    print('Parsing location...', end='')
    part = df['location'].str.split(",", expand=True)
    new = part[0].str.split(' ', expand=True)
    dot()

    df['lat'] = new[2].str.replace(')', '')
    df['lat'].astype(float)
    dot()

    df['long'] = new[1].str.replace('(', '')
    df['long'].astype(float)
    dot()

    df.drop(columns=['location'], inplace=True)
    print(' OK')


def remove_outliers(df, col):
    print('Remove OUTLIERS', end='')
    # Remove outliers. Outliers defined as values greater than 99.5th percentile
    max_val = np.percentile(df[col], 99.5)
    dot()
    df = df[df[col] <= max_val]
    print('.. OK')
    return df


def fix_priority(df_in):
    print('Fixing priority ', end='')
    df = df_in.copy()
    df.priority.replace(['A', 'B', 'C', 'D', 'E'], ['2', '2', '2', '3', '3'], inplace=True)
    dot()

    df = df[(df.priority != 'I') & (df.priority != '1')]
    df.priority.dropna(axis=0, inplace=True)
    dot()

    df.priority.replace(['A', 'B', 'C', 'E'], ['2', '2', '2', '3'], inplace=True)

    df.priority.astype(int)
    print('. OK')

    return df


def parser(df):
    # received_dt_tm,on_scene_dt_tm,available_dt_tm
    print('Parser', end='')
    df['rec_dt'] = pd.to_datetime(df['received_dt_tm'], format="%m/%d/%Y %I:%M:%S %p")
    # df['rec_dt_eu'] = df['rec_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    dot()

    df['onscene_dt'] = pd.to_datetime(df['on_scene_dt_tm'], format="%m/%d/%Y %I:%M:%S %p")
    # df['onscene_dt_eu'] = df['onscene_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    dot()

    df['end_dt'] = pd.to_datetime(df['available_dt_tm'], format="%m/%d/%Y %I:%M:%S %p")
    # df['end_dt_eu'] = df['end_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    print('. OK')


def feature_extraction(df):
    print('Feature extraction START', end='')
    # Extract date, month, hour of start
    df['rec_day'] = df['rec_dt'].dt.day
    dot()

    df['rec_month'] = df['rec_dt'].dt.month
    dot()

    df['rec_hour'] = df['rec_dt'].dt.hour
    dot()

    df['rec_day_of_week'] = df['rec_dt'].dt.weekday
    df['week'] = df['rec_dt'].dt.week
    df['year'] = df['rec_dt'].dt.year
    print(' OK')

    print('Feature extraction END', end='')
    # Extract date, month, hour of end
    df['end_day'] = df['end_dt'].dt.day
    dot()

    df['end_month'] = df['end_dt'].dt.month
    dot()

    df['end_hour'] = df['end_dt'].dt.hour
    dot()

    df['end_day_of_week'] = df['end_dt'].dt.weekday
    print(' OK')

    print('Feature extraction DURATION', end='')
    # Extract duration
    d = df['end_dt'] - df['rec_dt']
    dot()

    d = d / 1000000000
    dot()

    df['duration'] = d.astype(int)
    print('. OK')

    print('Feature extraction RESPONSE TIME', end='')
    # Extract duration
    d = df['onscene_dt'] - df['rec_dt']
    dot()

    d = d / 1000000000
    dot()

    df['res_time'] = d.astype(int)
    print('. OK')

    return df


def remove_nan(df):
    print('Removing NaN rows [' + str((df['end_dt'].isnull().sum() / df.size) * 100) + ']...', end='')
    df = df[np.isfinite(df['end_dt'])]
    print(' OK')

    print('Removing NaN rows [' + str((df['onscene_dt'].isnull().sum() / df.size) * 100) + ']...', end='')
    df = df[np.isfinite(df['onscene_dt'])]
    print(' OK')

    return df


def convert(df, dict_dest, col):
    temp_dict = dict(df[col])

    i = 0
    for t in temp_dict.values():
        if t not in dict_dest.keys():
            dict_dest[t] = i
            i = i + 1

    df[col].replace(dict_dest, inplace=True)


def station_area_location(df):
    print('Locating stations', end='')
    # Below the fire station addresses are scraped from the website.
    req = requests.get('https://sf-fire.org/fire-station-locations')
    soup = BeautifulSoup(req.content)
    addresses = soup.findAll('table')[0].findAll('tr')
    list_addresses = [[i.a for i in addresses][j].contents for j in range(len(addresses) - 1)]
    geolocator = Nominatim(timeout=60)

    list_addresses2 = [list_addresses[i][0].replace('\xa0', ' ').split(' at')[0] for i in range(len(list_addresses))]
    list_addresses3 = [(list_addresses2[i] + ', San Francisco') for i in range(len(list_addresses2))]
    dot()

    # Determine the coordinates of each address, and also
    # impute certain coordinates using Google
    geo_list = []
    [geo_list.append(geolocator.geocode(y)) for y in list_addresses3]
    geo_list = list(filter(None, geo_list))
    geo_list2 = [geo_list[i][1] for i in range(len(geo_list))]
    geo_list2.insert(3, (37.772780, -122.389050))  # 4
    geo_list2.insert(32, (37.794610, -122.393260))  # 35
    geo_list2.insert(42, (37.827160, -122.368300))  # 48
    del geo_list2[43]  # Delete incorrect rows
    del geo_list2[43]
    geo_list2.insert(44, (37.801640, -122.455553))  # 51
    geo_list2.insert(45, (37.622202, -122.3811767))  # A1
    geo_list2.insert(46, (37.622202, -122.3811767))  # A2
    geo_list2.insert(47, (37.622202, -122.3811767))  # A3
    geo_list2.insert(48, (37.622202, -122.3811767))
    dot()

    df2_2 = df.copy()
    df2_2 = df2_2.loc[(df2_2.station_area != '94') &
                      (df2_2.station_area != 'F3') &
                      (df2_2.station_area != 'E2') &
                      (df2_2.station_area != '47')]
    df2_2.dropna(subset=['station_area'], inplace=True)
    dot()

    stations_list = df2_2['station_area'].unique()

    stations_list = pd.DataFrame(np.sort(stations_list))
    station_capacity = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    starting_loc = pd.concat([stations_list, pd.DataFrame(geo_list2), pd.DataFrame(station_capacity)], axis=1)
    starting_loc.columns = ['station', 'lats', 'longs', 'stat_size']

    df2_2['station_size'] = df2_2['station_area'].map(starting_loc.set_index('station')['stat_size'])
    df2_2['station_lat'] = df2_2['station_area'].map(starting_loc.set_index('station')['lats'])
    df2_2['station_long'] = df2_2['station_area'].map(starting_loc.set_index('station')['longs'])

    df2_2.reset_index(drop=True, inplace=True)
    print(' OK')
    return df2_2


def corr_map(df):
    print('Correlation map', end='')
    corr = df.corr()
    dot()

    plt.figure(figsize=(18, 18))
    sns.heatmap(corr, cmap='seismic', annot=True, linewidths=0.2, vmin=-1, vmax=1, square=False)
    dot()

    plt.title('Correlation Map')
    plt.ylabel('Features')
    plt.xlabel('Features')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('plot/corr_map.png', dpi=300)
    dot()

    plt.show()
    print(' OK')


def distplot(df, col):
    print('Distribution of ' + col, end='')
    plt.figure(figsize=(12, 12))
    sub = df[(df[col] < 18000) & (df[col] > 0)]
    dot()

    sns.distplot(sub[col], kde=True)
    dot()

    plt.title('Distribution of ' + col)
    plt.ylabel('Density')
    plt.xlabel(col)
    dot()

    plt.minorticks_on()
    plt.show()
    print(' OK')


def weekday_hour(df_op):
    print('Hours distribution', end='')
    df = df_op[['rec_day_of_week', 'rec_hour', 'duration']]

    df_operations_day = pd.pivot_table(df[['rec_day_of_week', 'rec_hour', 'duration']],
                                       index=['rec_day_of_week', 'rec_hour'], aggfunc='count')
    df_operations_day = df_operations_day.sort_values(by=['rec_day_of_week'])
    dot()

    heatmap_data = pd.pivot_table(df_operations_day, values='duration', index='rec_day_of_week', columns='rec_hour')
    dot()

    plt.figure(figsize=(12, 5))
    sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.1, vmin=0, square=True,
                cbar_kws={"orientation": "horizontal"})
    dot()

    plt.title('Operations over Day/Hours')
    plt.ylabel('Weekday')
    plt.xlabel('Hours')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('plot/hours.png', dpi=300)
    plt.show()
    print(' OK')


def year_calendar(df_op):
    print('Years calendar', end='')
    df = df_op[['rec_day_of_week', 'week', 'priority']]

    df_operations_vehicle = pd.pivot_table(df, index=['rec_day_of_week', 'week'], aggfunc='count')
    dot()

    df_operations_vehicle2 = df_operations_vehicle.sort_values('rec_day_of_week', ascending=True)
    dot()

    heatmap_data = pd.pivot_table(df_operations_vehicle2, values='priority', columns='week', index='rec_day_of_week')
    plt.figure(figsize=(16, 4))
    sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.01, vmin=0, square=True,
                cbar_kws={"orientation": "horizontal"})
    dot()

    plt.title('Operations over Years')
    plt.ylabel('Weekdays')
    plt.xlabel('Week')
    plt.tight_layout()
    plt.savefig('plot/calendar.png', dpi=300)
    plt.show()
    print(' OK')


def op_over_month_station(df_op):
    print('Heatmap stations', end='')
    df = df_op[['station_area', 'rec_month', 'priority']]

    df_operations_day = pd.pivot_table(df, index=['station_area', 'rec_month'], aggfunc='count')
    dot()

    heatmap_data = pd.pivot_table(df_operations_day, values='priority', columns='station_area', index='rec_month')
    dot()

    plt.figure(figsize=(17, 6))
    sns.heatmap(heatmap_data, cmap="PuRd", linewidths=0.01, vmin=0, square=True, cbar_kws={"orientation": "horizontal"})
    dot()

    plt.title('Operations of different station over months')
    plt.xlabel('Station Area')
    plt.ylabel('Month')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('plot/month_station.png', dpi=300)
    plt.show()
    print(' OK')


def hier_clust(df):
    print('Hierarchical clustering', end='')
    dfc = df[['priority', 'call_type_group']].copy()
    dot()

    dfc = dfc.groupby('call_type_group')
    priority = dfc.priority.value_counts().unstack().fillna(0)
    dot()

    priority_normalized = priority.div(priority.sum(axis=1), axis=0)
    h_cluster = sns.clustermap(priority_normalized, annot=True, cmap='Reds', fmt='g')

    h_cluster.fig.suptitle("Hierarchical Clustering of Call Type Group vs. Priority", size=25)
    ax = h_cluster.ax_heatmap
    ax.set_xlabel('Priority Level')
    ax.set_ylabel('Call Type Group')
    plt.savefig('plot/hier_type-prior.png', dpi=300)
    plt.show()
    print(' OK')


def export_csv(df, path):
    print('Exporting to CSV [' + path + ']..', end='')
    df.to_csv('data/' + path + '.csv', index=False)
    print('. OK')


def replace_dict(df, dict, col):
    print('Replacing value with dictionary...', end='')
    df.replace({col: dict}, inplace=True)
    print(' OK')


def operations_map(df):
    print('Map of operations...', end='')
    plt.figure(figsize=(20, 20))
    red = df[df['year'] == 2018]
    sns.scatterplot(x=red['lat'], y=red['long'], hue=red['rec_day_of_week'])
    plt.title('Operation coordinates')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.minorticks_on()
    plt.tight_layout()
    plt.grid()
    plt.savefig('plot/coordinates.png', dpi=300)
    plt.show()
    print(' OK')


def year_month(df):
    group = df.groupby(['year', 'rec_month', 'rec_day'])['unit_id'].count()
    group.to_csv('sffd.csv', index=True)


def station_unit(df):
    group = pd.pivot_table(df[['station_area', 'unit_id', 'priority']], index=['station_area', 'unit_id'], aggfunc='count')
    heatmap_data = pd.pivot_table(group, values='priority', columns='unit_id', index='station_area')

    plt.figure(figsize=(30, 6))
    sns.heatmap(heatmap_data, cmap="PuRd", linewidths=0.01, vmin=0, square=True, cbar_kws={"orientation": "horizontal"})

    plt.title('Station-Unit')
    plt.xlabel('Unit ID')
    plt.ylabel('Station Area')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('plot/stationArea_units.png', dpi=300)


def op_unit(df_op):
    print('Heatmap unit', end='')
    df = df_op[['unit_id', 'rec_month', 'priority']]

    df_operations_day = pd.pivot_table(df, index=['unit_id', 'rec_month'], aggfunc='count')
    dot()

    heatmap_data = pd.pivot_table(df_operations_day, values='priority', columns='unit_id', index='rec_month')
    dot()

    plt.figure(figsize=(50,8))
    sns.heatmap(heatmap_data, cmap="PuRd", linewidths=0.01, vmin=0, square=True, cbar_kws={"orientation": "horizontal"})
    dot()

    plt.title('Operations of different unit over months')
    plt.xlabel('Unit ID')
    plt.ylabel('Month')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('plot/month_units.png', dpi=300)
    plt.show()
    print(' OK')


def main(path):
    t0 = time.time()
    print("Starting Analysis...")

    df = read_data(path)
    if 'RAW' in path:
        print("=== REDUCING DATASET ===\n")
        clean_col_names(df)
        df = data_reduction(df)
        location(df)
        df = station_area_location(df)
        export_csv(df, 'operationsSFFD_REDUCED')

    elif 'REDUCED' in path:
        print("=== CLEANING DATASET ===\n")
        parser(df)
        df = remove_nan(df)
        df = fix_priority(df)
        df = feature_extraction(df)
        df = data_reduction2(df)
        df = remove_outliers(df, 'duration')
        df = remove_outliers(df, 'res_time')
        export_csv(df, 'fd_data')
        replace_dict(df, typo, 'call_type')
        replace_dict(df, subtypo, 'call_type_group')
        export_csv(df, 'operationsSFFD_CLEANED')
        station_unit(df)
        # convert(df, unit, 'unit_id')
        # convert(df, stations, 'station_area')

    else:
        print("=== COMPUTING DATASET ===\n")
        # Plots
        corr_map(df)
        distplot(df, 'duration')
        distplot(df, 'res_time')
        weekday_hour(df)
        year_calendar(df)
        op_over_month_station(df)
        hier_clust(df)
        year_month(df)
        op_unit(df)
        station_unit(df)

        # REQUIRE SEABORN 0.9.0
        # operations_map(df)

    print("\nTotal Time [{} s]".format(round(time.time() - t0, 2)))


if __name__ == "__main__":
    main(sys.argv[1])
