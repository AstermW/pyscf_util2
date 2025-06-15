def parse_geometry(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    start_idx = None
    end_idx = None

    # Find the indices for the GEOM and NATOM sections
    for i, line in enumerate(lines):
        if line.strip() == "GEOM":
            start_idx = i + 1
        elif line.strip().startswith("NATOM"):
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        raise ValueError("GEOM or NATOM section not found in the file.")

    geometry = []
    for line in lines[start_idx:end_idx]:
        parts = line.split()
        if len(parts) >= 4:  # Ensure the line has at least atom and 3 coordinates
            atom = parts[0]
            coords = list(map(float, parts[1:4]))
            geometry.append(f"{atom} {coords[0]} {coords[1]} {coords[2]}")

    return "\n".join(geometry)


# Example usage:
# file_path = 'Untitled-1'
# geometry_string = parse_geometry(file_path)
# print(geometry_string)
