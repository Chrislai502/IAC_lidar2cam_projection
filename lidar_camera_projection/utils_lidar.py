import numpy as np
import math

# ---------------------------------------------------------------------------- #
#         Helper Functions for Projection Calculation and Visualization        #
# ---------------------------------------------------------------------------- #

# ------ Converts pointclouds(in camera frame) to spherical Coordinates ------ # (Spherical coordinates following Wikipedia definition here: https://en.wikipedia.org/wiki/Spherical_coordinate_system)
# Output points are in degrees
def xyz2spherical(ptc_arr):
    def cartesian_to_spherical(xyz):
        x, y, z = xyz # in camera frame
        rho   = math.sqrt(x**2 + y**2 + z**2)
        r     = math.sqrt(x**2 + y**2)
        
        # Determine theta
        if z>0:
            theta = math.atan(r/z)
        elif z<0:
            theta = math.pi + math.atan(r/z)
        elif z==0 and x!=0 and y!=0:
            theta = math.pi/2
        elif x==0 and y==0 and z==0:
            theta = None
        else:
            theta = math.acos(z/rho)
        
        # Determine Phi
        if x>0:
            phi = math.atan(y/x)
        elif x<0 and y>=0:
            phi = math.atan(y/x)
        elif x<0 and y<0:
            phi = math.atan(y/x) - math.pi
        elif x==0 and y>0:
            phi = math.pi
        elif x<0 and y<0:
            phi = math.pi
        elif x == 0 and y == 0:
            phi = None
        else:
            phi = (0 if y==0 else 1 if x > 0 else -1)*math.acos(x/r)

        return [phi, theta, rho]

    # Apply Spherical Conversion on each point
    return np.array(list(map(lambda x: cartesian_to_spherical(x), ptc_arr)))

# --------------- Converts Spherical Coordinates to Pointcloud in Camera Frame -------------- #
def spherical2xyz(spr_arr):
    def spherical_to_cartesian(spr):
        phi, theta, rho = spr
        x = rho * math.sin(theta) * math.cos(phi)
        y = rho * math.sin(theta) * math.sin(phi)
        z = rho * math.cos(theta)
        return [x, y, z]
    # Apply Cartesian Conversion on each point
    return np.array(list(map(lambda x: spherical_to_cartesian(x), spr_arr)))


# ---- Converts bbox coordinate format from (centerpoint, size_x, size_y) ---- #
# ------------------- to (top_y, bottom_y, left_x, right_x) ------------------ #
def box_to_corners(cx, cy, width, height):
    half_width = width / 2
    half_height = height / 2

    y1 = cy - half_height # top
    y2 = cy + half_height # bottom
    x1 = cx - half_width # left
    x2 = cx + half_width # right

    return (y1, y2, x1, x2) #(up, down, left, right)

# Helper function that convert list of boxes to a matrix
def boxes_to_matirx(boxes,offset=0): 
    '''
    Helper function that convert list of boxes to a matrix
    '''
    mat_return = []
    for box_msg in boxes:
        # print("Infunc: ", box_msg)
        top, bot, left, right = box_to_corners(box_msg.center.x, box_msg.center.y, box_msg.size_x, box_msg.size_y)
        mat = np.array([[right+offset, left-offset], 
                        [bot+offset,  top-offset], 
                        [1 , 1]]) # (2x3 camera corner matrix)
        # print("Infunc: ", mat)
        mat_return.append(mat)
    return np.stack(mat_return,axis=0)