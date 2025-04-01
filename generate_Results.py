# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 02:47:11 2023

@author: kaueu
"""

# Import necessary libraries for numerical operations, plotting, and data manipulation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

# Set the "choice" variable to select which analysis branch to run.
# Note: For consistency, when using the 2.5D model, the code always filters for Orientation == '2.5D'
choice = 2

if choice == 2:
    ##### If you want to run per AD Level (F-Measure by Orientation)
    # Read data from the consistent CSV file
    df = pd.read_csv('Nov1_fmeasure_hd_stages_scanner_final.csv', sep=',')
    # Drop rows with missing 'Stages' values and duplicate entries
    df = df.dropna(subset=['Stages'])
    df = df.drop_duplicates()
    
    # Create a boxplot of F-Measure across different Orientations and Styles
    plt.figure(figsize=(10, 4))
    sns.set_style('darkgrid')
    sns.boxplot(x='Orientation', y='F-Measure', hue='Style', data=df, showfliers=False)
    # Save the figure as a PDF
    plt.savefig('Fmeasure_ADlevel_Orientation.pdf', bbox_inches='tight')

elif choice == 2.1:
    ##### If you want to run per AD Level (IoU by Orientation)
    # Read data from the same CSV file
    df = pd.read_csv('Nov1_fmeasure_hd_stages_scanner_final.csv', sep=',')
    df = df.dropna(subset=['Stages'])
    df = df.drop_duplicates()
    
    # Create a boxplot of IoU across different Orientations and Styles
    plt.figure(figsize=(10, 4))
    sns.set_style('darkgrid')
    sns.boxplot(x='Orientation', y='IoU', hue='Style', data=df, showfliers=False)
    plt.savefig('IoU_ADlevel_Orientation.pdf', bbox_inches='tight')

elif choice == 2.2:
    ##### If you want to run per AD Level (Hausdorff distance by Orientation)
    # Read the data from the same CSV file
    df = pd.read_csv('Nov1_fmeasure_hd_stages_scanner_final.csv', sep=',')
    # Read voxel size information from a text file and compute the volume
    vox_size = pd.read_csv('voxel_size.txt', sep=' ')
    vox_size['V'] = vox_size['V_x'] * vox_size['V_y'] * vox_size['V_z']
    
    # Merge voxel volume with the main dataframe based on Subject and Dataset
    df = df.merge(vox_size[['Dataset', 'Subject', 'V']], how='left', on=['Subject', 'Dataset'])
    # Adjust Hausdorff distance by the voxel volume
    df['Hausdorff'] = df['Hausdorff'] * df['V']
    df = df.dropna(subset=['Stages'])
    df = df.drop_duplicates()
    
    # Create a boxplot of Hausdorff across different Orientations and Styles
    plt.figure(figsize=(10, 4))
    sns.set_style('darkgrid')
    sns.boxplot(x='Orientation', y='Hausdorff', hue='Style', data=df, showfliers=False)
    plt.savefig('Hausdorff_ADlevel_Orientation.pdf', bbox_inches='tight')

elif choice == 6.05:
    ##### Analysis for the 2.5D model: Vertical boxplot of F-Measure by AD Level and Style
    # Read data from the consistent CSV file and filter for valid 'Stages'
    df = pd.read_csv('Nov1_fmeasure_hd_stages_scanner_final.csv', sep=',')
    df = df.dropna(subset=['Stages']).drop_duplicates()
    # Ensure consistency by filtering for the 2.5D model only
    df = df[df['Orientation'] == '2.5D']
    
    # Create a taller figure for a vertical boxplot
    plt.figure(figsize=(6, 8))
    sns.set_style('darkgrid')
    # Plot F-Measure (x-axis) vs AD Level ('Stages', y-axis) split by Style
    sns.boxplot(y='Stages', x='F-Measure', hue='Style', data=df, showfliers=False)
    plt.xticks([])  # Remove x-axis tick labels for cleaner vertical layout
    plt.savefig('Fmeasure_ADlevel_Style_vertical.pdf', bbox_inches='tight')

elif choice == 6.1:
    ##### Analysis for the 2.5D model: IoU by AD Level and Style
    # Read data and filter for the 2.5D model
    df = pd.read_csv('Nov1_fmeasure_hd_stages_scanner_final.csv', sep=',')
    df = df.drop_duplicates()
    df = df[df['Orientation'] == '2.5D']
    
    plt.figure(figsize=(10, 4))
    sns.set_style('darkgrid')
    # Create a boxplot of IoU across different AD Levels ('Stages') split by Style
    sns.boxplot(x='Stages', y='IoU', hue='Style', data=df, showfliers=False)
    plt.ylim(0.55, 1)  # Set y-axis limits for better visualization
    plt.savefig('IoU_ADlevel_Style.pdf', bbox_inches='tight')

elif choice == 6.2:
    ##### Analysis for the 2.5D model: Hausdorff distance by AD Level and Style
    # Read data from the CSV file and remove duplicates
    df = pd.read_csv('Nov1_fmeasure_hd_stages_scanner_final.csv', sep=',')
    df = df.drop_duplicates()
    # Read voxel size information and compute the volume
    vox_size = pd.read_csv('voxel_size.txt', sep=' ')
    vox_size['V'] = vox_size['V_x'] * vox_size['V_y'] * vox_size['V_z']
    
    # Merge the voxel volume into the main dataframe
    df = df.merge(vox_size[['Dataset', 'Subject', 'V']], how='left', on=['Subject', 'Dataset'])
    # Adjust Hausdorff distance using the voxel volume
    df['Hausdorff'] = df['Hausdorff'] * df['V']
    
    plt.figure(figsize=(10, 4))
    sns.set_style('darkgrid')
    # For consistency, filter for the 2.5D model when plotting
    sns.boxplot(x='Stages', y='Hausdorff', hue='Style', data=df[df['Orientation'] == '2.5D'], showfliers=False)
    plt.savefig('Hausdorff_ADlevel_Style.pdf', bbox_inches='tight')

elif choice == 9:
    ##### Analysis for the 2.5D model: F-Measure by Scanner
    # Read data and filter for 2.5D Orientation
    df = pd.read_csv('Nov1_fmeasure_hd_stages_scanner_final.csv')
    df = df[df['Orientation'] == '2.5D']
    # Define a mapping for scanner names to a simplified code for consistent ordering
    scanner_mapping = {
        'GE Healthcare': 'A',
        'Siemens Prisma': 'B',
        'GE Signa': 'C',
        'Philips': 'D',
        'Siemens TrioTim': 'E'
    }
    # Map scanner names to the simplified codes
    df['Scanners'] = df['Scanner'].map(scanner_mapping)
    
    plt.figure(figsize=(10, 4))
    sns.set_style('darkgrid')
    # Create a boxplot of F-Measure by the mapped scanner codes, split by Style
    sns.boxplot(x='Scanners', y='F-Measure', hue='Style', data=df[df['Orientation'] == '2.5D'],
                showfliers=False, order=['A', 'B', 'C', 'D', 'E'])
    plt.savefig('F-Measure_Scanner.pdf', bbox_inches='tight')

elif choice == 9.1:
    ##### Analysis for the 2.5D model: IoU by Scanner
    df = pd.read_csv('Nov1_fmeasure_hd_stages_scanner_final.csv')
    df = df[df['Orientation'] == '2.5D']
    # Define scanner mapping as before
    scanner_mapping = {
        'GE Healthcare': 'A',
        'Siemens Prisma': 'B',
        'GE Signa': 'C',
        'Philips': 'D',
        'Siemens TrioTim': 'E'
    }
    df['Scanners'] = df['Scanner'].map(scanner_mapping)
    
    plt.figure(figsize=(10, 4))
    sns.set_style('darkgrid')
    # Create a boxplot of IoU by scanner code, split by Style
    sns.boxplot(x='Scanners', y='IoU', hue='Style', data=df, showfliers=False,
                order=['A', 'B', 'C', 'D', 'E'])
    plt.savefig('IoU_Scanner.pdf', bbox_inches='tight')

elif choice == 9.2:
    ##### Analysis for the 2.5D model: Hausdorff distance by Scanner (inconsistent branch note)
    df = pd.read_csv('Nov1_fmeasure_hd_stages_scanner_final.csv')
    df[df['Orientation'] == '2.5D']  # This line does not reassign; may be a mistake.
    vox_size = pd.read_csv('voxel_size.txt', sep=' ')
    vox_size['V'] = vox_size['V_x'] * vox_size['V_y'] * vox_size['V_z']
    
    df = df.merge(vox_size[['Dataset', 'Subject', 'V']], how='left', on=['Subject', 'Dataset'])
    df['Hausdorff'] = df['Hausdorff'] * df['V']
    df = df[df['Orientation'] == '2.5D']
    scanner_mapping = {
        'GE Healthcare': 'A',
        'Siemens Prisma': 'B',
        'GE Signa': 'C',
        'Philips': 'D',
        'Siemens TrioTim': 'E'
    }
    df['Scanners'] = df['Scanner'].map(scanner_mapping)
    plt.figure(figsize=(10, 4))
    sns.set_style('darkgrid')
    sns.boxplot(x='Scanners', y='Hausdorff', hue='Style', data=df, showfliers=False,
                order=['A', 'B', 'C', 'D', 'E'])
    plt.savefig('Hausdorff_Scanner.pdf', bbox_inches='tight')
