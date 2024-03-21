#!/bin/bash

export video=NYU_15thfloor

step2=$(sbatch step2_equa2pers.sbatch)
step2=${step2: -8}
step3=$(sbatch --dependency=afterok:$step2 step3_0_features.sbatch)
step3=${step3: -8}
step4=$(sbatch --dependency=afterok:$step3 step3_1_set_dataset.sbatch)
step4=${step4: -8}
step5=$(sbatch --dependency=afterok:$step4 step3_2_reconstruct.sbatch)
step5=${step5: -8}
sbatch --dependency=afterok:$step5 step4_0_s2c.sbatch

#step7=$(sbatch --dependency=afterok:$step5 step4_1_0_bundle.sbatch)
#step7=${step7: -8}
#step8=$(sbatch --dependency=afterok:$step7 step4_1_1_dense.sbatch)
#step8=${step8: -8}
#sbatch --dependency=afterok:$step8 step4_1_2_fuse.sbatch