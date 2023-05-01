import os

from ChexnetTrainer import ChexnetTrainer
from arguments import  parse_args


def main ():

    args = parse_args()

    try:
        os.mkdir(args.save_dir)
    except OSError as error:
        print(error)

    print("######"+args.data_root)
    trainer = ChexnetTrainer(args)
    print ('Testing the trained model')

    most_common = trainer.inference()
    CLASSES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                     'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    unseen_classes = ['Edema', 'Pneumonia', 'Emphysema', 'Fibrosis']

    for i in range(len(most_common)):
        if most_common[i] != -1:
            found_class = CLASSES[most_common[i]]
            if found_class in unseen_classes:
                print("UNSEEN=" + found_class)
            else:
                print("SEEN=" + found_class)

if __name__ == '__main__':
    main()