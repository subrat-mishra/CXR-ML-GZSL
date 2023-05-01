import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer
from arguments import  parse_args


def main ():

    args = parse_args()
    
    try:  
        os.mkdir(args.save_dir)  
    except OSError as error:
        print(error) 
    
    trainer = ChexnetTrainer(args)
    print ('Testing the trained model')
    

    test_ind_auroc, test_ind_auroc_seen, test_ind_auroc_unseen, precision_2, recall_2, f1_score_2, precision_3, recall_3, f1_score_3 = trainer.test()
    test_ind_auroc = np.array(test_ind_auroc)

    auroc_mean = np.array(test_ind_auroc).mean()
    auroc_seen_mean = np.array(test_ind_auroc_seen).mean()
    auroc_unseen_mean = np.array(test_ind_auroc_unseen).mean()

    print("######")
    print(f'AUROC S={auroc_seen_mean} U={auroc_unseen_mean} H={auroc_mean}')

    print(f'k=2: recall={recall_2}, precision={precision_2}, f1_score={f1_score_2}')
    print(f'k=3: recall={recall_3}, precision={precision_3}, f1_score={f1_score_3}')

    print(f'k=2: recall={recall_2:.2f}, precision={precision_2:.2f}, f1_score={f1_score_2:.2f}')
    print(f'k=3: recall={recall_3:.2f}, precision={precision_3:.2f}, f1_score={f1_score_3:.2f}')
    print("######")

    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')
            

if __name__ == '__main__':
    main()





