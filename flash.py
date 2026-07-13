#!/usr/bin/env python3
"""
Improved FLASH HDF5 to VTU converter that includes physical variables.
Handles the specific structure found in your FLASH file.
"""

import h5py
import numpy as np
import sys
import os

def examine_flash_file(filename):
    """Examine the structure of a FLASH HDF5 file."""
    print(f"Examining FLASH file: {filename}")
    print("=" * 50)
    
    with h5py.File(filename, 'r') as f:
        variables_found = []
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                shape = f[key].shape
                print(f"  DATASET: {key} - shape: {shape}, dtype: {f[key].dtype}")
                
                # Look for 4D arrays that could be variables [blocks, nx, ny, nz]
                if len(shape) == 4 and shape[0] > 1:
                    variables_found.append(key)
        
        # Check mesh information
        mesh_keys = ['coordinates', 'bounding box', 'block size', 'refine level']
        has_mesh = all(key in f for key in mesh_keys[:2])  # Need at least coordinates and bounding box
        
        if has_mesh:
            nblocks = f['coordinates'].shape[0]
            bbox_shape = f['bounding box'].shape
            print(f"\nMesh info: {nblocks} blocks, bbox shape: {bbox_shape}")
        
        print(f"\nFound {len(variables_found)} physical variables:")
        for var in variables_found:
            print(f"  - {var}")
        
        return {
            'has_mesh_data': has_mesh,
            'nblocks': nblocks if has_mesh else 0,
            'variables': variables_found,
            'bbox_shape': bbox_shape if has_mesh else None
        }

def create_vtu_with_variables(filename, output_file, max_blocks=None, include_vars=None):
    """Create a VTU file from FLASH data including physical variables."""
    
    with h5py.File(filename, 'r') as f:
        if not ('coordinates' in f and 'bounding box' in f):
            print("Required mesh data not found")
            return False
        
        # Read mesh data
        coords = f['coordinates'][:]
        bbox = f['bounding box'][:]
        
        nblocks = coords.shape[0]
        if max_blocks and nblocks > max_blocks:
            print(f"Limiting to first {max_blocks} blocks (out of {nblocks})")
            nblocks = max_blocks
            coords = coords[:max_blocks]
            bbox = bbox[:max_blocks]
        
        print(f"Processing {nblocks} blocks")
        
        # Grid size (16x16x16 from your file)
        nxb = nyb = nzb = 16
        
        # Get refinement levels
        refine_level = None
        if 'refine level' in f:
            refine_level = f['refine level'][:nblocks]
        
        # Read variable data
        variables = {}
        if include_vars is None:
            # Use common FLASH variables found in your file
            include_vars = ['dens', 'pres', 'tele', 'tion', 'depo']
        
        for var_name in include_vars:
            if var_name in f:
                print(f"Reading variable: {var_name}")
                variables[var_name] = f[var_name][:nblocks, :, :, :]
            else:
                print(f"Variable {var_name} not found in file")
        
        # Write VTU file
        write_vtu_with_variables(bbox, nxb, nyb, nzb, output_file, nblocks, 
                                variables, refine_level)
        
        return True

def write_vtu_with_variables(bbox, nxb, nyb, nzb, output_file, nblocks, variables, refine_level=None):
    """Write VTU file including physical variables."""
    
    print(f"Writing VTU file with variables: {output_file}")
    
    total_points = nblocks * (nxb + 1) * (nyb + 1) * (nzb + 1)
    total_cells = nblocks * nxb * nyb * nzb
    
    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{total_points}" NumberOfCells="{total_cells}">\n')
        
        # Write points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        
        for block in range(nblocks):
            # Handle bounding box format: [block, dim, min/max]
            xmin, xmax = bbox[block, 0, 0], bbox[block, 0, 1]
            ymin, ymax = bbox[block, 1, 0], bbox[block, 1, 1]
            zmin, zmax = bbox[block, 2, 0], bbox[block, 2, 1]
            
            # Create uniform grid within block
            dx = (xmax - xmin) / nxb
            dy = (ymax - ymin) / nyb
            dz = (zmax - zmin) / nzb
            
            for k in range(nzb + 1):
                for j in range(nyb + 1):
                    for i in range(nxb + 1):
                        x = xmin + i * dx
                        y = ymin + j * dy
                        z = zmin + k * dz
                        f.write(f'          {x:.6e} {y:.6e} {z:.6e}\n')
        
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        
        # Write cells
        f.write('      <Cells>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        
        for block in range(nblocks):
            base_point = block * (nxb + 1) * (nyb + 1) * (nzb + 1)
            
            for k in range(nzb):
                for j in range(nyb):
                    for i in range(nxb):
                        base = base_point + k * (nxb+1) * (nyb+1) + j * (nxb+1) + i
                        
                        vertices = [
                            base,                                   # 0
                            base + 1,                              # 1
                            base + (nxb+1) + 1,                   # 2
                            base + (nxb+1),                       # 3
                            base + (nxb+1)*(nyb+1),               # 4
                            base + (nxb+1)*(nyb+1) + 1,           # 5
                            base + (nxb+1)*(nyb+1) + (nxb+1) + 1, # 6
                            base + (nxb+1)*(nyb+1) + (nxb+1)      # 7
                        ]
                        
                        f.write('          ' + ' '.join(map(str, vertices)) + '\n')
        
        f.write('        </DataArray>\n')
        
        # Write offsets
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        for i in range(1, total_cells + 1):
            f.write(f'          {i * 8}\n')
        f.write('        </DataArray>\n')
        
        # Write cell types
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for i in range(total_cells):
            f.write('          12\n')  # VTK_HEXAHEDRON = 12
        f.write('        </DataArray>\n')
        f.write('      </Cells>\n')
        
        # Write cell data
        f.write('      <CellData>\n')
        
        # Block ID
        f.write('        <DataArray type="Int32" Name="BlockID" format="ascii">\n')
        for block in range(nblocks):
            for cell in range(nxb * nyb * nzb):
                f.write(f'          {block}\n')
        f.write('        </DataArray>\n')
        
        # Refinement level
        if refine_level is not None:
            f.write('        <DataArray type="Int32" Name="RefinementLevel" format="ascii">\n')
            for block in range(nblocks):
                level = refine_level[block]
                for cell in range(nxb * nyb * nzb):
                    f.write(f'          {level}\n')
            f.write('        </DataArray>\n')
        
        # Physical variables
        for var_name, var_data in variables.items():
            print(f"Writing variable: {var_name}")
            f.write(f'        <DataArray type="Float32" Name="{var_name}" format="ascii">\n')
            for block in range(nblocks):
                for k in range(nzb):
                    for j in range(nyb):
                        for i in range(nxb):
                            value = var_data[block, i, j, k]
                            f.write(f'          {value:.6e}\n')
            f.write('        </DataArray>\n')
        
        f.write('      </CellData>\n')
        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python flash_to_vtu_with_vars.py <input.h5> [output.vtu] [options]")
        print("")
        print("Options:")
        print("  --examine           Just examine the file structure")
        print("  --max-blocks N      Limit number of blocks to process")
        print("  --vars var1,var2    Specific variables to include (default: dens,pres,tele,tion,depo)")
        print("")
        print("Examples:")
        print("  python flash_to_vtu_with_vars.py nif.flash --examine")
        print("  python flash_to_vtu_with_vars.py nif.flash output.vtu --max-blocks 100")
        print("  python flash_to_vtu_with_vars.py nif.flash output.vtu --vars dens,pres,tion")
        return 1
    
    input_file = sys.argv[1]
    
    # Parse arguments
    examine_only = '--examine' in sys.argv
    max_blocks = None
    include_vars = None
    
    if '--max-blocks' in sys.argv:
        idx = sys.argv.index('--max-blocks')
        if idx + 1 < len(sys.argv):
            max_blocks = int(sys.argv[idx + 1])
    
    if '--vars' in sys.argv:
        idx = sys.argv.index('--vars')
        if idx + 1 < len(sys.argv):
            include_vars = sys.argv[idx + 1].split(',')
    
    # Determine output file
    if len(sys.argv) >= 3 and not sys.argv[2].startswith('--'):
        output_file = sys.argv[2]
    else:
        output_file = os.path.splitext(input_file)[0] + '_with_vars.vtu'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return 1
    
    try:
        file_info = examine_flash_file(input_file)
        
        if not examine_only:
            if file_info['has_mesh_data']:
                print("\nStarting conversion with variables...")
                success = create_vtu_with_variables(input_file, output_file, max_blocks, include_vars)
                
                if success:
                    print(f"\nConversion completed!")
                    print(f"Output file: {output_file}")
                    print("You can now open this file in ParaView to visualize the data and variables.")
                else:
                    print("Conversion failed")
                    return 1
            else:
                print("Cannot convert: Required mesh data not found")
                return 1
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())