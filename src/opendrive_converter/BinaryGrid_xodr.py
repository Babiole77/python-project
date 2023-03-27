import os
import sys
import csv
import math
import platform
import subprocess
import numpy as np
from math import nan
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon

# Global Constant variables
H_SCALE = 10
text_x_offset = 0
text_y_offset = 0.7
text_size = 7
esmini_track_file = 'track.csv'
parking_slots_file = 'Parking_Slot.csv'
parking_slots_column = ["X0", "Y0", "Z0", "X1", "Y1", "Z1", "X2", "Y2", "Z2", "X3", "Y3", "Z3", "Parking Slot Type",
                        "Occupation Status", "Object_ID"]
object_data_file = 'object_data.csv'
object_data_csv_column = ["Object_ID", "Object_Name", "Object_Type", "Dynamic_Object", "X0", "Y0", "Z0", "X1", "Y1",
                          "Z1", "X2", "Y2", "Z2", "X3", "Y3", "Z3"]
occ_map_file = 'OCC_MAP_Export.yaml'
default_xodr_file = 'APA_Perpend_Parking_Reverse_Park_SC_2.xodr'
# Global calibration variables
resolution = 0.1
plot_graph = True
# Below variables value are in units: Meters
default_park_width = 3
default_park_length = 6
rd_marking_width = 0.15
parked_obj_adjstd_todetct = 1.2
parked_obj_offset_adj_x = 0
park_slot_offset_adj_x = 1
ego_start_x = 30
ego_start_y = -1.75
ego_width = 1.632
ego_length = 4.426
fig, axs = plt.subplots(2, 1)


def main():
    xodr_file = check_xodr_file_exists(sys.argv, default_xodr_file)
    road_coordinates = parse_xodr_by_esmini(xodr_file, esmini_track_file)
    lane_x, lane_y, border_x, border_y, ref_x, ref_y = normalize_road_coordinates(road_coordinates)
    xml_parse_data = read_xml_file(xodr_file)
    final_ego_start_x, final_ego_start_y = ego_veh_start_pos_calc(xodr_file, ref_x, ref_y, ego_start_x, ego_start_y)
    x_min, x_max, y_min, y_max = calc_min_max_lane_x_y(lane_x, lane_y, border_x, border_y)
    grid_x, grid_y = calc_grid_size(x_min, x_max, y_min, y_max)
    binary_grid = gen_binary_grid(x_min, y_min, grid_x, grid_y, road_coordinates)
    export_binary_grid_yaml(binary_grid, xodr_file, grid_x, grid_y, resolution,
                            (final_ego_start_x + abs(x_min)) / resolution,
                            (final_ego_start_y + abs(y_min)) / resolution, occ_map_file)
    all_roads = xml_parse_data.findAll('road')
    object_parked, parking_lanes = parse_parked_objects(all_roads)
    object_data = calc_object_coord(road_coordinates, x_min, y_min, ref_x, ref_y, all_roads, object_parked)
    parking_slots_data = calc_parking_slot_coord(parking_lanes, road_coordinates, x_min, y_min, object_data)
    export_to_csv(object_data_file, parking_slots_column, object_data)
    export_to_csv(parking_slots_file, parking_slots_column, parking_slots_data)
    print("********** Program Execution Successful **********")
    if plot_graph:
        axs[1].add_patch(Rectangle(((final_ego_start_x + abs(x_min)) / resolution,
                                    (final_ego_start_y + abs(y_min) - (ego_width / 2)) / resolution),
                                   ego_length / resolution, ego_width / resolution, linewidth=1, edgecolor='violet',
                                   facecolor='violet', fill=True, zorder=10))
        axs[0].set_aspect('equal', 'datalim')
        axs[1].imshow(binary_grid, cmap='binary', interpolation='none', extent=(0, grid_x, 0, grid_y))
        plt.show()


def export_to_csv(csv_file_name, columns, to_csv_data):
    """export to_csv_data variable array data to csv

    Args:
        csv_file_name (string): target csv file name
        columns (array): header text of csv file in array
        to_csv_data (2d array): csv array data
    """
    with open(csv_file_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(columns)
        writer.writerows(to_csv_data)
    print("********** " + csv_file_name + " file is Exported **********")


def check_xodr_file_exists(user_input_arg, default_xodr_file):
    """Check xodr file exist or not if not Exit program

    Args:
        user_input_arg (string): file name as system argument
        default_xodr_file (string): default file name

    Returns:
        string: verified xodr file name
    """
    xodr_file = user_input_arg[1] if len(user_input_arg) > 1 else default_xodr_file
    if (os.path.isfile(xodr_file)):
        print("********** Step(1/12) " + xodr_file + " file Found **********")
        return xodr_file
    else:
        print("********** Step(1/12) " + xodr_file + " file Not Found **********")
        print("Exiting the Program")
        sys.exit(0)


def parse_xodr_by_esmini(xodr_file, esmini_track_file):
    """parse the xodr file using esmini library to plot 2d map

    Args:
        xodr_file (string): xodr file name with path

    Returns:
        road_coordinates: 2d road coordinates of all lanes
    """
    if platform.system() == "Windows":
        cmd = "odrplot.exe " + xodr_file
        with open('nul', 'w') as devnull:
            subprocess.check_call(cmd, stdout=devnull, stderr=subprocess.STDOUT)
    elif platform.system() == "Linux":
        os.system("chmod +x odrplot")
        cmd = "odrplot " + xodr_file
        with open('/dev/null', 'w') as devnull:
            subprocess.check_call(cmd, stdout=devnull, stderr=subprocess.STDOUT)
    else:
        print("The operating system is not Windows or Linux. Exiting the Program")
        sys.exit(0)
    print("********** Step(2/12) track.csv is Generated Successfully **********")
    road_coordinates = read_csv_file(esmini_track_file)
    print("********** Step(3/12) Reading Track.csv file **********")
    return road_coordinates


def read_csv_file(csv_file):
    """read data from csv file

    Args:
        csv_file (string): csv file name with path

    Returns:
        csv_data: csv file read data as 2d array
    """
    with open(csv_file) as f:
        reader = csv.reader(f, skipinitialspace=True)
        csv_data = list(reader)
    return csv_data


def read_xml_file(xml_file):
    """read data from xml file

    Args:
        xml_file (string): xml file name with path

    Returns:
        xml_parse_data: xml file read data as object
    """
    with open(xml_file, 'r') as f:
        data = f.read()
    xml_parse_data = BeautifulSoup(data, "xml")
    print("********** Step(5/12) Parsed " + xml_file + " file **********")
    return xml_parse_data


def normalize_road_coordinates(road_coordinates):
    """Road coordniates adjusted with absolute reference value.
        if more than one road definition available, adjusting
        the coordinates based on reference value of ref_x,ref_y

    Args:
        road_coordinates (2d array): lane coordinates on 2d grid/map

    Returns:
        lane_x: 1d array of x axis values of lane
        lane_y: 1d array of y axis values of lane
        border_x: 1d array of x axis values of border lane
        border_y: 1d array of y axis values of border lane
        ref_x: 1d array of x axis values of center or reference line of road
        ref_y: 1d array of y axis values of center or reference line of road
    """
    ref_x, ref_y, ref_z, ref_h = [], [], [], []
    lane_x, lane_y, lane_z, lane_h = [], [], [], []
    border_x, border_y, border_z, border_h = [], [], [], []
    road_id, road_id_x, road_id_y = [], [], []
    road_start_dots_x, road_start_dots_y = [], []
    lane_section_dots_x, lane_section_dots_y = [], []
    arrow_dx, arrow_dy = [], []
    current_road_id, current_lane_id, current_lane_section = None, None, None
    for i in range(len(road_coordinates) + 1):
        if i < len(road_coordinates):
            pos = road_coordinates[i]
        if i == len(road_coordinates) or (pos[0] == 'lane' and i > 0 and current_lane_id == '0'):
            if current_lane_section == '0':
                road_id.append(int(current_road_id))
                index = int(len(ref_x[-1]) / 3.0)
                h = ref_h[-1][index]
                road_id_x.append(ref_x[-1][index] + (text_x_offset * math.cos(h) - text_y_offset * math.sin(h)))
                road_id_y.append(ref_y[-1][index] + (text_x_offset * math.sin(h) + text_y_offset * math.cos(h)))
                road_start_dots_x.append(ref_x[-1][0])
                road_start_dots_y.append(ref_y[-1][0])
                if len(ref_x) > 0:
                    arrow_dx.append(ref_x[-1][1] - ref_x[-1][0])
                    arrow_dy.append(ref_y[-1][1] - ref_y[-1][0])
                else:
                    arrow_dx.append(0)
                    arrow_dy.append(0)
            lane_section_dots_x.append(ref_x[-1][-1])
            lane_section_dots_y.append(ref_y[-1][-1])
        if i == len(road_coordinates):
            break
        if pos[0] == 'lane':
            current_road_id = pos[1]
            current_lane_section = pos[2]
            current_lane_id = pos[3]
            if pos[3] == '0':
                ltype = 'ref'
                ref_x.append([])
                ref_y.append([])
                ref_z.append([])
                ref_h.append([])
            elif pos[4] == 'no-driving':
                ltype = 'border'
                border_x.append([])
                border_y.append([])
                border_z.append([])
                border_h.append([])
            else:
                ltype = 'lane'
                lane_x.append([])
                lane_y.append([])
                lane_z.append([])
                lane_h.append([])
        else:
            if ltype == 'ref':
                ref_x[-1].append(float(pos[0]))
                ref_y[-1].append(float(pos[1]))
                ref_z[-1].append(float(pos[2]))
                ref_h[-1].append(float(pos[3]))
            elif ltype == 'border':
                border_x[-1].append(float(pos[0]))
                border_y[-1].append(float(pos[1]))
                border_z[-1].append(float(pos[2]))
                border_h[-1].append(float(pos[3]))
            else:
                lane_x[-1].append(float(pos[0]))
                lane_y[-1].append(float(pos[1]))
                lane_z[-1].append(float(pos[2]))
                lane_h[-1].append(float(pos[3]))
    if plot_graph:
        MainPlot_2d = plt.figure(1)
        # plot road ref line segments
        for i in range(len(ref_x)):
            axs[0].plot(ref_x[i], ref_y[i], linewidth=2.0, color='#BB5555')
        # plot driving lanes in blue
        for i in range(len(lane_x)):
            axs[0].plot(lane_x[i], lane_y[i], linewidth=1.0, color='#3333BB')
        # plot border lanes in gray
        for i in range(len(border_x)):
            axs[0].plot(border_x[i], border_y[i], linewidth=1.0, color='#AAAAAA')
        # plot red dots indicating lane dections
        for i in range(len(lane_section_dots_x)):
            axs[0].plot(lane_section_dots_x[i], lane_section_dots_y[i], 'o', ms=4.0, color='#BB5555')
        for i in range(len(road_start_dots_x)):
            # plot a yellow dot at start of each road
            axs[0].plot(road_start_dots_x[i], road_start_dots_y[i], 'o', ms=5.0, color='#BBBB33')
            # and an arrow indicating road direction
            axs[0].arrow(road_start_dots_x[i], road_start_dots_y[i], arrow_dx[i], arrow_dy[i], width=0.1,
                         head_width=1.0, color='#BB5555')
        # plot road id numbers
        for i in range(len(road_id)):
            axs[0].text(road_id_x[i], road_id_y[i], road_id[i], size=text_size, ha='center', va='center',
                        color='#222222')
    print("********** Step(4/12) Road coordinates Normalized **********")
    return lane_x, lane_y, border_x, border_y, ref_x, ref_y


def ego_veh_start_pos_calc(xodr_file, ref_x, ref_y, lc_ego_start_x, lc_ego_start_y):
    """calculate ego vehicle start position based on
        i. if .rd5 file available then parse from rd5 file
        ii. if .rd5 not available then take default values
            from lc_ego_start_x, lc_ego_start_y

    Args:
        xodr_file (string): xodr file name
        ref_x (1d array): x coordiantes to the center of road 
        ref_y (1d array): y coordiantes to the center of road 
        lc_ego_start_x (float): ego start position
        lc_ego_start_y (float): ego start position

    Returns:
        lc_ego_start_x: x coordinate of Ego vehicle start position
        lc_ego_start_y: y coordinate of Ego vehicle start position
    """
    file_name = os.path.basename(xodr_file)
    if (os.path.isfile(str(file_name[:-5]) + ".rd5")):
        with open((str(file_name[:-5]) + ".rd5")) as f:
            reader = csv.reader(f, skipinitialspace=True)
            road_coordinates = list(reader)
        start_pos_read = False
        print("********** (Optional) rd5 File Found **********")
        for rc in road_coordinates:
            if 'UserPath.0.Nodes:' in rc:
                start_pos_read = True
            elif start_pos_read:
                start_pos_read = False
                values = rc[0].strip().split()
                lc_ego_start_x = float(values[2])
                for rx in range(0, len(ref_x[0])):
                    if float(ref_x[0][rx]) == lc_ego_start_x:
                        lc_ego_start_y = float(values[4]) + float(ref_y[0][rx])
                        break
        return lc_ego_start_x, lc_ego_start_y
    else:
        print("********** (Optional) rd5 File Not Found **********")
        return lc_ego_start_x, lc_ego_start_y


def calc_min_max_lane_x_y(lane_x, lane_y, border_x, border_y):
    """calculate min and max value of lane_x array and border_x array
        and calculate min and max value of lane_y array and border_y array

    Args:
        lane_x (1d array): lane x coordinates
        lane_y (1d array): lane y coordinates
        border_x (1d array): border x coordinates
        border_y (1d array): border y coordinates

    Returns:
        x_min: minimum value of x from lane_x and border_x
        x_max: maximum value of y from lane_y and border_y
        y_min: minimum value of x from lane_x and border_x
        y_max: maximum value of y from lane_y and border_y
    """
    maxY = ((max(max(x) for x in lane_y)) if len(lane_y) > 0 else 0,
            (max(max(x) for x in border_y)) if len(border_y) > 0 else 0)
    minY = ((min(min(x) for x in lane_y)) if len(lane_y) > 0 else 0,
            (min(min(x) for x in border_y)) if len(border_y) > 0 else 0)
    y_max = (max(maxY))
    y_min = (min(minY))
    maxX = ((max(max(x) for x in lane_x)) if len(lane_x) > 0 else 0,
            (max(max(x) for x in border_x)) if len(border_x) > 0 else 0)
    minX = ((min(min(x) for x in lane_x)) if len(lane_x) > 0 else 0,
            (min(min(x) for x in border_x)) if len(border_x) > 0 else 0)
    x_max = (max(maxX))
    x_min = (min(minX))
    print("********** Step(6/12) Minimum and maximum range of X and Y coordinates are calculated **********")
    return x_min, x_max, y_min, y_max


def calc_grid_size(x_min, x_max, y_min, y_max):
    """caluclate grid size by shifting the range to postive axis
        for easy iteration

    Args:
        x_min: minimum value of x from lane_x and border_x
        x_max: maximum value of y from lane_y and border_y
        y_min: minimum value of x from lane_x and border_x
        y_max: maximum value of y from lane_y and border_y

    Returns:
        grid_x: grid size on x axis
        grid_y: grid size on y axis
    """
    grid_x = int(round((abs(x_min) + abs(x_max)) / resolution, 0))
    grid_y = int(round((abs(y_min) + abs(y_max)) / resolution, 0))
    print("********** Step(7/12) Grid size is calculated **********")
    return grid_x, grid_y


def gen_binary_grid(x_min, y_min, grid_x, grid_y, road_coordinates):
    """iterate through 2d coordinates to identify road and fill occupancy data

    Args:
        x_min: minimum value of x from lane_x and border_x
        y_min: minimum value of x from lane_x and border_x
        grid_x: grid size on x axis
        grid_y: grid size on y axis
        road_coordinates (2d array): lane coordinates on 2d map

    Returns:
        binary_grid: numpy array having binary occupancy data
    """
    PreviousLane = nan
    PreviousSubLane = 0
    StartLaneX, EndLaneX = [], []
    FillStart = False
    binary_grid = np.zeros((grid_y, grid_x), dtype=bool)
    for pos in road_coordinates:
        if pos[0] == 'lane':
            if (math.isnan(PreviousLane)):
                FillStart = True
                StartLaneX = []
                EndLaneX = []
            else:
                If_Same_Lane = (int(pos[1]) == PreviousLane) and (int(pos[3]) != PreviousSubLane) and len(
                    EndLaneX) > 0 and len(StartLaneX) > 0
                If_Different_Lane = (int(pos[1]) != PreviousLane) and len(EndLaneX) > 0 and len(StartLaneX) > 0
                if (int(pos[1]) == PreviousLane) and (int(pos[3]) != PreviousSubLane) and len(EndLaneX) == 0:
                    FillStart = False
                elif If_Same_Lane or If_Different_Lane:
                    binary_grid = fill_binary_grid(binary_grid, StartLaneX, EndLaneX, grid_y)
                    if If_Same_Lane:
                        StartLaneX = EndLaneX
                        EndLaneX = []
                        FillStart = False
                    elif If_Different_Lane:
                        FillStart = True
                        StartLaneX = []
                        EndLaneX = []
            PreviousLane = int(pos[1])
            PreviousSubLane = int(pos[3])
        elif FillStart:
            StartLaneX.append([PreviousLane, PreviousSubLane, int(round((float(pos[0]) + abs(x_min)) / resolution, 0)),
                               int(round((float(pos[1]) + abs(y_min)) / resolution, 0))])
        elif not FillStart:
            EndLaneX.append([PreviousLane, PreviousSubLane, int(round((float(pos[0]) + abs(x_min)) / resolution, 0)),
                             int(round((float(pos[1]) + abs(y_min)) / resolution, 0))])
    print("********** Step(8/12) binary_grid is generated **********")
    binary_grid = fill_binary_grid(binary_grid, StartLaneX, EndLaneX, grid_y)
    return binary_grid


def fill_binary_grid(binary_grid, StartLaneX, EndLaneX, grid_y):
    """filling binary grid numpy array with True when road is detected

    Args:
        binary_grid (numpy array): numpy array having binary occupancy data
        StartLaneX (2d array): starting index array of road
        EndLaneX (2d array): ending index array of road
        grid_y (float): grid size on y axis

    Returns:
        binary_grid: numpy array having binary occupancy data
    """
    if max(len(StartLaneX), len(EndLaneX)) == 1:
        i = 1
        y_indices = grid_y - np.arange(min(StartLaneX[i - 1][3], EndLaneX[i - 1][3]),
                                       max(StartLaneX[i - 1][3], EndLaneX[i - 1][3])) - 1
        x_indices = np.arange(min(StartLaneX[i - 1][2], EndLaneX[i - 1][2]), min(StartLaneX[i][2], EndLaneX[i][2]))
        binary_grid[y_indices[:, np.newaxis], x_indices] = True
    else:
        for i in range(1, max(len(StartLaneX), len(EndLaneX))):
            y_indices = grid_y - np.arange(min(StartLaneX[i - 1][3], EndLaneX[i - 1][3]),
                                           max(StartLaneX[i - 1][3], EndLaneX[i - 1][3])) - 1
            x_indices = np.arange(min(StartLaneX[i - 1][2], EndLaneX[i - 1][2]), min(StartLaneX[i][2], EndLaneX[i][2]))
            binary_grid[y_indices[:, np.newaxis], x_indices] = True
    return binary_grid


def export_binary_grid_yaml(binary_grid, xodr_file, grid_x, grid_y, resolution, lc_ego_start_x, lc_ego_start_y,
                            occ_map_file):
    """export binary grid data to yaml file for ROS to read occupancy status

    Args:
        binary_grid (numpy array): numpy array having binary occupancy data
        xodr_file (string): xodr file name
        grid_x (float): grid size on x axis
        grid_y (float): grid size on y axis
        resolution (float): resolution of the grid size
        lc_ego_start_x (float): x axis coordinates for ego vehicle start position
        lc_ego_start_y (float): y axis coordinates for ego vehicle start position

    Returns:
        Nothing
    """
    OCCData = binary_grid * 100
    with open(occ_map_file, 'w') as f:
        f.write('map:')
        f.write('\n name: "' + str(xodr_file[:-5]) + '"')
        f.write('\n size_x: ' + str(grid_x))
        f.write('\n size_y: ' + str(grid_y))
        f.write('\n resolution: ' + str(resolution))
        f.write('\n vehicle_start_x: ' + str(lc_ego_start_x))
        f.write('\n vehicle_start_y: ' + str(lc_ego_start_y))
        f.write('\n data:')
        for ODI in range(len(OCCData) - 1, -1, -1):
            data_bytes = ('\n - [' + ', '.join(map(str, OCCData[ODI])) + ']')
            f.write(data_bytes)
    print("********** Step(9/12) binary_grid is Exported to " + occ_map_file + " **********")
    return True


def parse_parked_objects(all_roads):
    """parse all objects on the road from xml parsed data

    Args:
        all_roads (object): all road definistions in array objects

    Returns:
        object_parked: array of object coordinates on the binary grid
        parking_lanes: array of all parking lane details from roads definition
    """
    object_parked = []
    parking_lanes = []
    for i in range(len(all_roads)):
        object_parked.append([int(all_roads[i].get('id')), all_roads[i].findAll('object')])
    for i in range(len(all_roads)):
        if len(all_roads[i].findAll('lane', {'type': 'parking'})) > 0:
            parking_lanes.append([int(all_roads[i].get('id')), all_roads[i].findAll('lane', {'type': 'parking'}),
                                  float(all_roads[i].get('length'))])
    print("********** Step(10/12) Object data is Parsed from XODR file **********")
    return object_parked, parking_lanes


def calc_object_coord(road_coordinates, x_min, y_min, ref_x, ref_y, all_roads, object_parked):
    """calculate object coordinates on the grid map

    Args:
        road_coordinates (2d array): road coordinates 2d coordinates
        x_min (float): grid x axis minimum value
        y_min (float): grid y axis minimum value
        ref_x (array): x coordiantes to the center of road 
        ref_y (array): y coordiantes to the center of road 
        all_roads (road object): road object data from xodr
        object_parked (parked object): object parked data from xodr

    Returns:
        _type_: _description_
    """
    obj_coords = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", "", "NO", False]
    object_data, Lane_IDs = [], []
    Road_ID, Lane_ID = 0, 0
    X_Adjust = 0.0
    Lane_IDs = [[int(LID.get("id")) for LID in all_roads[i].findAll('lane')] for i in range(len(all_roads)) if
                len(all_roads[i].findAll('lane')) > 0]
    for Obj in object_parked:
        for i in range(len(road_coordinates)):
            if road_coordinates[i][0] == 'lane':
                Road_ID = int(road_coordinates[i][1])
                Lane_ID = int(road_coordinates[i][3])
                X_Adjust = float(road_coordinates[i + 1][0])
                Min_LaneID = min(Lane_IDs[Road_ID])
                Max_LaneID = max(Lane_IDs[Road_ID])
            elif (Obj[0] == Road_ID):
                for DObj in Obj[1]:
                    Repeat_Data = DObj.findAll("repeat")
                    if len(Repeat_Data) > 0:
                        Repeat_Data_s = Repeat_Data[0].get("s")
                        Repeat_Data_len = Repeat_Data[0].get("length")
                        Repeat_Data_dis = Repeat_Data[0].get("distance")
                    DObj_t = DObj.get("t")
                    DObjid = DObj.get("id")
                    DObjname = DObj.get("name")
                    Allparameters = DObj.findAll('parameters')
                    for parameters in Allparameters:
                        if (parameters.get("s0")):
                            S0 = parameters.get("s0")
                            S1 = parameters.get("s1")
                            pminX = parameters.get("minX")
                            pmaxX = parameters.get("maxX")
                            pRotZ = parameters.get("rotZ")
                            Starting_X = (abs(x_min) + X_Adjust + float(S0))
                            if (road_coordinates[i - 1][0] != 'lane') and (
                                    float(road_coordinates[i - 1][0]) <= Starting_X <= float(road_coordinates[i][0])):
                                Ref_Idx = ref_x[Road_ID].index(float(road_coordinates[i][0]))
                                Ref_Y_Adjust = ref_y[Road_ID][Ref_Idx]
                                if ((Lane_ID < 0) and (float(DObj_t) < 0) and (Lane_ID == Min_LaneID)) or (
                                        (Lane_ID >= 0) and (float(DObj_t) >= 0) and (Lane_ID == Max_LaneID)):
                                    if (len(Repeat_Data) > 0) and (abs(float(pminX)) + abs(float(pmaxX))) > (
                                            float(S1) - float(S0)):
                                        obj_coords = calc_obj_rect_coord(parameters, x_min, X_Adjust, y_min,
                                                                         Ref_Y_Adjust, DObjid, DObjname, float(S0))
                                    elif (len(Repeat_Data) > 0):
                                        Start_Point = abs(x_min) + X_Adjust + float(Repeat_Data_s)
                                        End_Point = Start_Point + abs(x_min) + X_Adjust + float(Repeat_Data_len)
                                        Point_Gap = float(Repeat_Data_dis)
                                        for cp in np.arange(Start_Point, End_Point, Point_Gap):
                                            obj_coords = calc_obj_rect_coord(parameters, x_min, X_Adjust, y_min,
                                                                             Ref_Y_Adjust, DObjid, DObjname, cp)
                                            FObjdata = check_rotation_append(obj_coords[12], obj_coords[13],
                                                                             obj_coords[14], obj_coords[15],
                                                                             float(pRotZ), obj_coords)
                                            object_data.append(FObjdata) if FObjdata not in object_data else object_data
                                    elif (float(S0) == float(S1)) and (len(Repeat_Data) <= 0):
                                        obj_coords = calc_obj_rect_coord(parameters, x_min, X_Adjust, y_min,
                                                                         Ref_Y_Adjust, DObjid, DObjname, float(S0))
                                if obj_coords[16]:
                                    FObjdata = check_rotation_append(obj_coords[12], obj_coords[13], obj_coords[14],
                                                                     obj_coords[15], float(pRotZ), obj_coords)
                                    object_data.append(FObjdata) if FObjdata not in object_data else object_data
                                    obj_coords[16] = False
    print("********** Step(11/12) Objects Coordinates are calculated **********")
    return object_data


def calc_obj_rect_coord(parameters, x_min, X_Adjust, y_min, Ref_Y_Adjust, DObjid, DObjname, s0):
    """calculate object rectangle coordinates on grid map

    Args:
        parameters (xml object): object parameters from xml
        x_min (float): grid x axis minimum value
        X_Adjust (float): x axis offset to match absolute length of road
        y_min (float): grid y axis minimum value
        Ref_Y_Adjust (float): y axis offset to match absolute width of road
        DObjid (int): object unique ID
        DObjname (string): object name as per xodr
        s0 (float): starting point of object w.r.t road length

    Returns:
        array: rectangle coordinates and object details
    """
    pminX = parameters.get("minX")
    pmaxX = parameters.get("maxX")
    pt = parameters.get("t")
    pminY = parameters.get("minY")
    pmaxY = parameters.get("maxY")
    pminZ = parameters.get("minZ")
    pmaxZ = parameters.get("maxZ")
    pgeoObjFile = parameters.get("geoObjFile")
    ObjX0 = float(pminX) + abs(x_min) + X_Adjust + s0 + parked_obj_offset_adj_x
    ObjX1 = float(pmaxX) + abs(x_min) + X_Adjust + s0 + parked_obj_offset_adj_x
    ObjX2 = ObjX1
    ObjX3 = ObjX0
    ObjY0 = abs(y_min) + float(pminY) + float(pt) + Ref_Y_Adjust
    ObjY1 = ObjY0
    ObjY2 = abs(y_min) + float(pmaxY) + float(pt) + Ref_Y_Adjust
    ObjY3 = ObjY2
    ObjZ0 = float(pminZ) + float(pmaxZ)
    ObjZ1 = ObjZ0
    ObjZ2 = ObjZ0
    ObjZ3 = ObjZ0
    Object_ID = int(DObjid)
    Object_Name = DObjname
    Object_Type = (pgeoObjFile.split('/'))[1]
    Dynamic_Object = "NO"
    Object_Detected = True
    return ObjX0, ObjX1, ObjX2, ObjX3, ObjY0, ObjY1, ObjY2, ObjY3, ObjZ0, ObjZ1, ObjZ2, ObjZ3, Object_ID, Object_Name, Object_Type, Dynamic_Object, Object_Detected


def check_rotation_append(Object_ID, Object_Name, Object_Type, Dynamic_Object, rotZ, obj_coords):
    """check if object has rotated by its access by rotZ variable

    Args:
        Object_ID (int): unique ID for object
        Object_Name (string): Object name as per xodr
        Object_Type (string): object type/classification as per xodr
        Dynamic_Object (string): is dynamically movable object Yes or No
        rotZ (float): rotation in radians unit
        obj_coords (array): object coordinates

    Returns:
        FObjdata: array of Finalized and identified objects in array
    """
    ObjX0 = obj_coords[0]
    ObjY0 = obj_coords[1]
    ObjX1 = obj_coords[2]
    ObjY1 = obj_coords[3]
    ObjX2 = obj_coords[4]
    ObjY2 = obj_coords[5]
    ObjX3 = obj_coords[6]
    ObjY3 = obj_coords[7]
    ObjZ0 = obj_coords[8]
    ObjZ1 = obj_coords[9]
    ObjZ2 = obj_coords[10]
    ObjZ3 = obj_coords[11]
    if float(rotZ) != 0:
        Rad = float(rotZ)
        centerX = (ObjX0 + ObjX3) / 2
        centerY = (ObjY1 + ObjY2) / 2
        angle_radians = -Rad
        rot_matrix = np.array(
            [[np.cos(angle_radians), -np.sin(angle_radians)], [np.sin(angle_radians), np.cos(angle_radians)]])
        rect_coords = np.array([[ObjX0, ObjY0],
                                [ObjX1, ObjY1],
                                [ObjX2, ObjY2],
                                [ObjX3, ObjY3]])
        rect_coords -= np.array([centerX, centerY])
        rect_coords = np.dot(rect_coords, rot_matrix)
        rect_coords += np.array([centerX, centerY])
        rectangle = Polygon([(rect_coords[0, 0] / resolution, rect_coords[0, 1] / resolution),
                             (rect_coords[1, 0] / resolution, rect_coords[1, 1] / resolution),
                             (rect_coords[2, 0] / resolution, rect_coords[2, 1] / resolution),
                             (rect_coords[3, 0] / resolution, rect_coords[3, 1] / resolution)], color='green',
                            zorder=10)
        axs[1].add_patch(rectangle)
        FObjdata = [Object_ID, Object_Name, Object_Type, Dynamic_Object, rect_coords[0, 0] / resolution,
                    rect_coords[0, 1] / resolution, ObjZ0 / resolution, rect_coords[1, 0] / resolution,
                    rect_coords[1, 1] / resolution, ObjZ1 / resolution, rect_coords[2, 0] / resolution,
                    rect_coords[2, 1] / resolution, ObjZ2 / resolution, rect_coords[3, 0] / resolution,
                    rect_coords[3, 1] / resolution, ObjZ3 / resolution]
    else:
        axs[1].add_patch(
            Rectangle((ObjX0 / resolution, ObjY0 / resolution), abs(ObjX0 / resolution - ObjX1 / resolution),
                      abs(ObjY3 / resolution - ObjY1 / resolution), linewidth=1, edgecolor='green', facecolor='green',
                      fill=True, zorder=10))
        FObjdata = [Object_ID, Object_Name, Object_Type, Dynamic_Object, ObjX0 / resolution, ObjY0 / resolution,
                    ObjZ0 / resolution, ObjX1 / resolution, ObjY1 / resolution, ObjZ1 / resolution, ObjX2 / resolution,
                    ObjY2 / resolution, ObjZ2 / resolution, ObjX3 / resolution, ObjY3 / resolution, ObjZ3 / resolution]
    return FObjdata


def calc_parking_slot_coord(parking_lanes, road_coordinates, x_min, y_min, object_data):
    """calculate parking slot coordinates based on parking lane dimensions

    Args:
        parking_lanes (object): parking lane object details
        road_coordinates (2d array): road coordinates 2d coordinates
        x_min (float): minimum x value before shift
        y_min (float): minimum y value before shift
        object_data (_type_): all object detail list in 2d array

    Returns:
        parking_slots_data: 2d array of parking slot data
    """
    parking_slots_data = []
    for pl in range(len(parking_lanes)):
        P_Road_Id = int(parking_lanes[pl][0])
        for lane in range(len(parking_lanes[pl][1])):
            P_Lane_Id = int(parking_lanes[pl][1][lane].get('id'))
            Parking_Area = parking_lanes[pl][1][lane].findAll('width')
            Width_Data = []
            Del_Arr = []
            for j in range(len(Parking_Area)):
                Width_Data.append([float(Parking_Area[j].get('sOffset')), float(Parking_Area[j].get('a')),
                                   float(Parking_Area[j].get('c')), P_Road_Id, P_Lane_Id])
            if len(Width_Data) > 1:
                for wd in range(1, len(Width_Data) - 1):
                    if (Width_Data[wd][1] == Width_Data[wd - 1][1]) and (
                            Width_Data[wd][2] == Width_Data[wd - 1][2]) and (
                            Width_Data[wd][3] == Width_Data[wd - 1][3]) and (
                            Width_Data[wd][4] == Width_Data[wd - 1][4]) and (
                            Width_Data[wd][1] == Width_Data[wd + 1][1]) and (
                            Width_Data[wd][2] == Width_Data[wd + 1][2]) and (
                            Width_Data[wd][3] == Width_Data[wd + 1][3]) and (
                            Width_Data[wd][4] == Width_Data[wd + 1][4]):
                        Del_Arr.append(wd)
            for del_idx in Del_Arr[::-1]:
                del Width_Data[del_idx]
            if len(Parking_Area) > 0:
                for w in range(len(Width_Data)):
                    Start_Point = float(Width_Data[w][0])
                    End_Point = float(Width_Data[w + 1][0]) if (w < len(Width_Data) - 1) else float(
                        parking_lanes[pl][2])
                    Length_Val = float(Width_Data[w][1])
                    Par_Count = 0.0
                    Current_Point = 0.0
                    Lane_Started = False
                    if (Length_Val != 0 and float(Width_Data[w][2]) == 0.0):
                        for pos in range(len(road_coordinates)):
                            Occup_Status = "Free"
                            Object_ID = 0
                            colorn = 'blue'
                            if (road_coordinates[pos][0] == 'lane') and (
                                    int(road_coordinates[pos][1]) == P_Road_Id) and (
                                    int(road_coordinates[pos][3]) == P_Lane_Id):
                                X_Adjust = float(road_coordinates[pos + 1][0])
                                Lane_Started = True
                                Par_Count = 1.0
                                Current_Point = X_Adjust
                            elif (road_coordinates[pos][0] == 'lane') and (
                                    (int(road_coordinates[pos][1]) != P_Road_Id) or (
                                    int(road_coordinates[pos][3]) != P_Lane_Id)):
                                Lane_Started = False
                            elif Lane_Started:
                                Park_length = default_park_width if (Length_Val >= 5) else default_park_length
                                if (float(Start_Point + X_Adjust + rd_marking_width) / resolution <= (
                                        ((abs(x_min) + Current_Point) / resolution) + (
                                        Par_Count * rd_marking_width / resolution))) and ((((
                                                                                                    abs(x_min) + Current_Point) / resolution) + (
                                                                                                   (
                                                                                                           Park_length) / resolution) + (
                                                                                                   Par_Count * rd_marking_width / resolution)) <= float(
                                    End_Point + X_Adjust + park_slot_offset_adj_x) / resolution):
                                    x0 = ((abs(x_min) + Current_Point) / resolution) + (
                                            Par_Count * rd_marking_width / resolution)
                                    x1 = ((abs(x_min) + Current_Point) / resolution) + (
                                            Par_Count * rd_marking_width / resolution) + (
                                                 (Park_length) / resolution)
                                    x2 = ((abs(x_min) + Current_Point) / resolution) + (
                                            Par_Count * rd_marking_width / resolution) + (
                                                 (Park_length) / resolution)
                                    x3 = ((abs(x_min) + Current_Point) / resolution) + (
                                            Par_Count * rd_marking_width / resolution)
                                    y0 = ((abs(y_min) + float(road_coordinates[pos][1])) / resolution)
                                    y1 = ((abs(y_min) + float(road_coordinates[pos][1])) / resolution)
                                    y2 = ((abs(y_min) + float(road_coordinates[pos][1])) / resolution) + (
                                            (float(Width_Data[w][1])) / resolution)
                                    y3 = ((abs(y_min) + float(road_coordinates[pos][1])) / resolution) + (
                                            (float(Width_Data[w][1])) / resolution)
                                    z = 0.0
                                    Current_Point = (Current_Point + Park_length)
                                    Par_Count = Par_Count + 1
                                    for OD in object_data:
                                        OBD_XMin = min(OD[4], OD[7], OD[10], OD[13]) + (
                                                parked_obj_adjstd_todetct / resolution)
                                        OBD_XMax = max(OD[4], OD[7], OD[10], OD[13]) - (
                                                parked_obj_adjstd_todetct / resolution)
                                        OBD_YMin = min(OD[5], OD[8], OD[11], OD[14]) + (
                                                parked_obj_adjstd_todetct / resolution)
                                        OBD_YMax = max(OD[5], OD[8], OD[11], OD[14]) - (
                                                parked_obj_adjstd_todetct / resolution)
                                        if (x0 <= OBD_XMin) and (x1 >= OBD_XMax) and (y0 <= OBD_YMin) and (
                                                y2 >= OBD_YMax):
                                            Occup_Status = "Occupied"
                                            colorn = 'yellow'
                                            Object_ID = OD[0]
                                            break
                                    if P_Lane_Id < 0:
                                        if plot_graph:
                                            axs[1].add_patch(
                                                Rectangle((x0, y0), abs(x0 - x1), abs(y2 - y1), linewidth=1,
                                                          edgecolor='r', facecolor=colorn, fill=True))
                                        parking_slots_data.append([x0, y0, z, x1, y1, z, x2, y2, z, x3, y3, z,
                                                                   "Perpendicular" if (Length_Val >= 5) else "Parallel",
                                                                   Occup_Status, Object_ID])
                                    else:
                                        # y0 -> y2; y1 -> y3; x2 -> x3
                                        if plot_graph:
                                            axs[1].add_patch(
                                                Rectangle((x0, y0), abs(x3 - x1), abs(y2 - y1), linewidth=1,
                                                          edgecolor='r', facecolor=colorn, fill=True))
                                        parking_slots_data.append([x0, y2, z, x1, y3, z, x3, y0, z, x2, y1, z,
                                                                   "Perpendicular" if (Length_Val >= 5) else "Parallel",
                                                                   Occup_Status, Object_ID])
                                else:
                                    Current_Point = Current_Point + 1
    print("********** Step(12/12) Parking Slots Coordinates are Calculated **********")
    return parking_slots_data


if __name__ == "__main__":
    main()
