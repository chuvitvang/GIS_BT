import numpy as np

def read_asc(file_path):
    """ Đọc file DEM .asc và trả về ma trận độ cao """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    metadata = [float(lines[i].split()[1]) for i in range(6)]
    ncols, nrows, xllcorner, yllcorner, cellsize, nodata = map(int, metadata[:2]) + metadata[2:]
    data = np.loadtxt(lines[6:], dtype=float)
    data[data == nodata] = np.nan
    return data, ncols, nrows, cellsize, nodata

def compute_flow_direction(elevation):
    """ Tính hướng dòng chảy D8 """
    D8_CODES = np.array([[32, 64, 128], [16, 0, 1], [8, 4, 2]])
    flow_direction = np.zeros_like(elevation, dtype=int)
    
    for i in range(1, elevation.shape[0] - 1):
        for j in range(1, elevation.shape[1] - 1):
            if np.isnan(elevation[i, j]): continue
            window = elevation[i-1:i+2, j-1:j+2]
            min_pos = np.unravel_index(np.nanargmin(window), window.shape)
            flow_direction[i, j] = D8_CODES[min_pos]
    
    return flow_direction

def compute_flow_accumulation(elevation, flow_direction):
    """ Tính tích lũy dòng chảy """
    flow_accumulation = np.zeros_like(elevation, dtype=int)
    
    for i in range(1, elevation.shape[0] - 1):
        for j in range(1, elevation.shape[1] - 1):
            if np.isnan(elevation[i, j]): continue
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < elevation.shape[0] and 0 <= nj < elevation.shape[1]:
                    if flow_direction[ni, nj] == D8_CODES[di + 1, dj + 1]:
                        flow_accumulation[i, j] += 1
    
    return flow_accumulation

# Đọc dữ liệu DEM
dem_file = "dem.asc"
elevation, ncols, nrows, cellsize, nodata_value = read_asc(dem_file)

# Tính toán hướng dòng chảy và tích lũy dòng chảy
flow_direction = compute_flow_direction(elevation)
flow_accumulation = compute_flow_accumulation(elevation, flow_direction)

# Xuất kết quả
print("📌 Ma trận độ cao:")
print(elevation)
print("✅ Flow Direction (D8):")
print(flow_direction)
print("✅ Flow Accumulation:")
print(flow_accumulation)
