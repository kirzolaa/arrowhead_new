Potential Plot Organization
=======================

1. Original Triple Points (in ./original/ directory):
   - Point 1: R0 = [ 0.05  -0.025 -0.025]
   - Point 2: R0 = [-0.025  0.05  -0.025]
   - Point 3: R0 = [-0.025 -0.025  0.05 ]

2. Nested Triple Points (in ./nested/ directory):
   - Point 1: R0 = [ 0.05816497 -0.02908248 -0.02908248] (nested around original point 1)
   - Point 2: R0 = [ 0.04591752 -0.01683503 -0.02908248] (nested around original point 1)
   - Point 3: R0 = [ 0.04591752 -0.02908248 -0.01683503] (nested around original point 1)
   - Point 4: R0 = [-0.02908248  0.05816497 -0.02908248] (nested around original point 2)
   - Point 5: R0 = [-0.02908248  0.04591752 -0.01683503] (nested around original point 2)
   - Point 6: R0 = [-0.01683503  0.04591752 -0.02908248] (nested around original point 2)
   - Point 7: R0 = [-0.02908248 -0.02908248  0.05816497] (nested around original point 3)
   - Point 8: R0 = [-0.01683503 -0.02908248  0.04591752] (nested around original point 3)
   - Point 9: R0 = [-0.02908248 -0.01683503  0.04591752] (nested around original point 3)

Each directory contains the following plot types for each point:
   - Va_Vx_R0_N.png: Combined plot of Va and Vx vs theta
   - Va_minus_Vx_R0_N.png: Plot of Va-Vx vs theta
   - Vx_minus_Va_R0_N.png: Plot of Vx-Va vs theta
   - all_potentials_R0_N.png: Combined plot of all potential relationships
