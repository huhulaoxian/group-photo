import os
import csv

def main():
    img_path = "../../data/compare"
    csv_path = "../../data/image_compare.csv"

    filelist = os.listdir(img_path)
    i = 1
    with open(csv_path, 'w',newline='') as f:
        for item in filelist:
            print(item)
            wr = csv.writer(f)
            wr.writerow([i,item])
            i=i+1


if __name__ == "__main__":
    main()
