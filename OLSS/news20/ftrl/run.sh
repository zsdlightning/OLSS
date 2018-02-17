#Choose the best parameters for FTRL on validation dataset
./tune_ftrl.sh
./tune_ftrl_l2.sh 1 0.005 L2

#use the best parameter for training and testing
./get_final_models.sh
