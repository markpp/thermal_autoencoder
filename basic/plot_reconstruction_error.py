import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # automatically (randomly) highlight regions with different levels of activity or manually


    #folder = "/home/markpp/github/thermal_autoencoder/output/202002071635 - 32of76/view1/"
    #frame_min, frame_max = 320, 1500
    #frames_of_interest = [360-frame_min,535-frame_min,944-frame_min,1386-frame_min]

    #folder = "/home/markpp/github/thermal_autoencoder/output/202002172020 - 3of64/view1/"
    #frame_min, frame_max = 600, 3000
    #frames_of_interest = [768-frame_min,944-frame_min,1386-frame_min,2463-frame_min]

    folder = "/home/markpp/github/thermal_autoencoder/output/202002172020 - 36of64/view1/"
    frame_min, frame_max = 250, 1500
    frames_of_interest = [354-frame_min,940-frame_min,1105-frame_min,1386-frame_min]


    losses = np.load(os.path.join(folder,'crop0_losses.npy'))[frame_min:frame_max]

    images = np.load(os.path.join(folder,'crop0_inputs.npy'))[frame_min:frame_max]

    recs = np.load(os.path.join(folder,'crop0_recs.npy'))[frame_min:frame_max]

    names = np.load(os.path.join(folder,'crop0_files.npy'))[frame_min:frame_max]

    frame_numbers = list(range(losses.shape[0]))

    plt.figure(figsize=(12,4))
    plt.plot(frame_numbers,losses,'--')
    plt.xlabel('frame number', fontsize=14)
    plt.ylabel('reconstruction error (MSE)', fontsize=14)
    for frame_num in frames_of_interest:
        img = images[frame_num]
        rec = recs[frame_num]
        name = os.path.basename(names[frame_num])
        cv2.imwrite(os.path.join(folder,name),img)
        name = name.replace('.jpg','_rec.jpg')
        cv2.imwrite(os.path.join(folder,name),rec)
        plt.axvline(x=frame_num,color='r',linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(folder,'reconstruction_error.png'))
    #plt.show()
