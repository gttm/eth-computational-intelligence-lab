# Patchwise Road Segmentation for Aerial Images with CNN
# Emmanouil Angelis, Spyridon Angelopoulos, Georgios Touloupas
# Group 5: Google Maps Team
# Department of Computer Science, ETH Zurich, Switzerland
# Computational Intelligence Lab

# This script is used generate the plots shown in the report

import argparse
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--losses_dir", type=str, default="experiment_logfiles", 
                        help="directory containing the losses of the models")
    args = parser.parse_args()
    
    steps = []
    losses = []
    #files = ["EXTENDED-56","EXTENDED-56-DROP-0.5", "EXTENDED-56-DROP-0.7","EXTENDED-56-DROP-0.9"]
    #files = ["EXTENDED-36","EXTENDED-56", "EXTENDED-76","EXTENDED-92"]
    files = ["VGG", "VGG-PRETRAINED", "VGG-BATCHNORM", "VGG-RESIDUAL"]
    models = files
    for file in files:
        steps_model = []
        losses_model = []
        filename = "{}/{}.txt".format(args.losses_dir, file)
        with open(filename, "r", encoding="utf8") as file:
            for line in file:
                if line.startswith("epoch:"):
                    tokens = line.strip().split(",")
                    steps_model.append(int( tokens[1].strip().split(" ")[1]))
                    losses_model.append(float(tokens[3].strip().split(" ")[1]))
                    print(tokens[3].strip().split(" ")[1])
                    
        steps.append(steps_model)
        losses.append(losses_model)
    print("models: " + str(models))
    print("steps: " + str(steps))
    print("losses: " + str(losses))

    max_step = 110000
    ticks = range(10000, max_step, 20000)
    colors = [(27,158,119), (217,95,2), (117,112,179), (231,41,138), (102,166,30), (230,171,2), (31,120,180)]
    #colors = [(166,206,227), (31,120,180), (178,223,138), (51,160,44), (251,154,153), (227,26,28), (253,191,111), (255,127,0), (202,178,214), (106,61,154), (255,255,153), (177,89,40)]
    colors = [(r/255.0, g/255.0, b/255.0) for r, g, b in colors]
    lines = ["-"]#["-.", ":", "-", "--"]
    
    plt.figure()
    for i in range(len(models)):
        if i==0 or i==1:
            plt.plot(steps[i][13:len(steps[i])-8], losses[i][13:len(losses[i])-8], linestyle=lines[i%len(lines)], color=colors[i%len(colors)], label=str(models[i]), linewidth=0.8)  
        else: 
            plt.plot(steps[i][13:], losses[i][13:], linestyle=lines[i%len(lines)], color=colors[i%len(colors)], label=str(models[i]), linewidth=0.8)
    plt.legend(loc="best")
    plt.xticks(ticks)
    plt.xlim(xmax=max_step)
    plt.grid()
    plt.xlabel("Training Steps")
    plt.ylabel("Validation Loss")
    plt.savefig(args.losses_dir + "/lossesSmallExperiments.png", dpi=1000)