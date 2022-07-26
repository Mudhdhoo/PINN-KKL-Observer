import torch

"""
Trained models of different systems

"""

RevDuff = {
    'model1':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/RevDuff/Seed 8/50ic_1000N_-1_1_15ep_SEED_lr0.001_bs32_scheduler' ),
    
    'model2': torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/RevDuff/Seed 8/50ic_1000N_-1_1_15ep_SEED_NoPde_lr0.001_bs32_scheduler'),
    
    'model3':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/RevDuff/Seed 9/50ic_1000N_-1_1_15ep_SEED_lr0.001_bs32_scheduler'),
    
    'model4':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/RevDuff/Seed 9/50ic_1000N_-1_1_15ep_SEED_NoPde_lr0.001_bs32_scheduler')
}

RevDuff_NA = {
    'model1':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/RevDuff_NA/Seed 5/50ic_1000N_-1_1_20ep_SEED_NoPde_lr0.0001_bs32_scheduler_my=1'  ),

    'model2':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/RevDuff_NA/Seed 5/50ic_1000N_-1_1_20ep_SEED_lr0.0001_bs32_scheduler_my=1'  )
}

VdP = {
    'model1':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/VdP/Seed 3/50ic_1000N_-1_1_15ep_SEED_lr0.001_bs32_scheduler' ),
    
    'model2':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/VdP/Seed 3/50ic_1000N_-1_1_15ep_SEED_NoPde_lr0.001_bs32_scheduler' ),
    
    'model3':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/VdP/Seed 4/50ic_1000N_-1_1_15ep_SEED_lr0.001_bs32_scheduler'),
    
    'model4':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/VdP/Seed 4/50ic_1000N_-1_1_15ep_SEED_NoPde_lr0.001_bs32_scheduler')
}

VdP_NA = {
    'model1':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/VdP_NA/Seed 5/50ic_1000N_-1_1_15ep_SEED_NoPde_lr0.0001_bs32_scheduler_my=1'  ),

    'model2':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/VdP_NA/Seed 5/50ic_1000N_-1_1_15ep_SEED_lr0.0001_bs32_scheduler_my=1'  )
}


Polynomial = {
    'model1':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/Polynomial/Seed 8/50ic_1000N_-1_1_15ep_SEED_NoPde_lr0.001_bs32_scheduler' ),
    
    'model2':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/Polynomial/Seed 8/50ic_1000N_-1_1_15ep_SEED_lr0.001_bs32_scheduler' )
}


Chua = {
    'model1': torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/Chua/Seed 9/50ic_1000N_-1_1_15ep_SEED_lr0.001_bs32_scheduler'  ),
    
    'model2': torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/Chua/Seed 9/50ic_1000N_-1_1_15ep_SEED_NoPde_lr0.001_bs32_scheduler'  ),
    
    'model3':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/Chua/Seed 9/50ic_1000N_-1_1_15ep_SEED_NoPde_lr0.0001_bs32_scheduler' ),
    
    'model4':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/Chua/Seed 9/50ic_1000N_-1_1_15ep_SEED_lr0.0001_bs32_scheduler' )
}

Rossler = {
    'model1':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/Rossler/Seed 1/50ic_1000N_-1_1_30ep_SEED_NoPde_lr0.001_bs32_scheduler'),
    
    'model2':torch.load('/Applications/Programming/Machine Learning/DF Internship/Models/Rossler/Seed 1/50ic_1000N_-1_1_15ep_SEED_lr0.0001_bs32_scheduler')

}










