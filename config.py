import argparse

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_path', type=str, default='/ssd_scratch/cvit/hwaseem04/writer/dataset/', help='specify data directory ending with "/"')
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--train_size', type=float, default=0.8)
        self.parser.add_argument('--patch_size', type=int, default=16, help='Patch size for patches')
        self.parser.add_argument('--epochs', type=int, default=5)
        self.parser.add_argument('--model_path', type=str, default="/")
        self.parser.add_argument('--csv_path', type=str, default="/ssd_scratch/cvit/hwaseem04/writer/dataset/val.csv", help="Path to test data csv")
        self.parser.add_argument('--test_path', type=str, default="/ssd_scratch/cvit/hwaseem04/writer/dataset/val", help="Path to test data")
    def parse(self):
        return self.parser.parse_args()

