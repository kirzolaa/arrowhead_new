#!/bin/bash

# Create a backup directory
mkdir -p backup_images

# Copy original files to backup
cp -r figures/* backup_images/

# Rename files
cd figures

# Distance effect files
if [ -f "d_effect_R0_0_0_0.png" ]; then
    cp d_effect_R0_0_0_0.png d_effect.png
fi

if [ -f "d_effect_R0_1_1_1.png" ]; then
    cp d_effect_R0_1_1_1.png d_effect_custom1.png
fi

if [ -f "d_effect_R0_0_0_2.png" ]; then
    cp d_effect_R0_0_0_2.png d_effect_custom2.png
fi

# Theta effect files
if [ -f "theta_effect_R0_0_0_0.png" ]; then
    cp theta_effect_R0_0_0_0.png theta_effect.png
fi

if [ -f "theta_effect_R0_1_1_1.png" ]; then
    cp theta_effect_R0_1_1_1.png theta_effect_custom1.png
fi

if [ -f "theta_effect_R0_0_0_2.png" ]; then
    cp theta_effect_R0_0_0_2.png theta_effect_custom2.png
fi

# Combined effect files
if [ -f "combined_effect_R0_0_0_0.png" ]; then
    cp combined_effect_R0_0_0_0.png combined_effect.png
fi

if [ -f "combined_effect_R0_1_1_1.png" ]; then
    cp combined_effect_R0_1_1_1.png combined_effect_custom1.png
fi

if [ -f "combined_effect_R0_0_0_2.png" ]; then
    cp combined_effect_R0_0_0_2.png combined_effect_custom2.png
fi

# Origin projections files
if [ -f "r0_projections_combined_effect_R0_0_0_0.png" ]; then
    cp r0_projections_combined_effect_R0_0_0_0.png origin_projections_combined_effect.png
fi

if [ -f "r0_projections_combined_effect_R0_1_1_1.png" ]; then
    cp r0_projections_combined_effect_R0_1_1_1.png origin_projections_combined_effect_custom1.png
fi

if [ -f "r0_projections_combined_effect_R0_0_0_2.png" ]; then
    cp r0_projections_combined_effect_R0_0_0_2.png origin_projections_combined_effect_custom2.png
fi

# Combined view files
if [ -f "combined_view_R0_0_0_0_d_1p5_theta_0p79.png" ]; then
    cp combined_view_R0_0_0_0_d_1p5_theta_0p79.png combined_view_default.png
fi

if [ -f "combined_view_R0_1_1_1_d_1p5_theta_0p79.png" ]; then
    cp combined_view_R0_1_1_1_d_1p5_theta_0p79.png combined_view_custom1.png
fi

if [ -f "combined_view_R0_0_0_2_d_1p5_theta_0p79.png" ]; then
    cp combined_view_R0_0_0_2_d_1p5_theta_0p79.png combined_view_custom2.png
fi

echo "Image renaming complete."
