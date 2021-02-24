import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
from string import Formatter

historyFilePath = open('./model/output/Attempt7/historyFile.pickle', "rb")

historyFile = pickle.load(historyFilePath)
historyFilePath.close()

epochs=[10*i for i in range(len(historyFile))]

time_2d = sum([step['unet_2d_time'] for step in historyFile], timedelta())
testDSC_2d = [step['testDSC_2d'] for step in historyFile]
loss_2d = [step['history_2d']['loss'][-1] for step in historyFile]
dice_coef_2d = [step['history_2d']['dice_coef'][-1] for step in historyFile]
mse_2d = [step['history_2d']['mse'][-1] for step in historyFile]
mae_2d = [step['history_2d']['mae'][-1] for step in historyFile]
acc_2d = [step['history_2d']['acc'][-1] for step in historyFile]
generalized_dice_coeff_2d = [step['history_2d']['generalized_dice_coeff'][-1] for step in historyFile]
val_loss_2d = [step['history_2d']['val_loss'][-1] for step in historyFile]
val_dice_coef_2d = [step['history_2d']['val_dice_coef'][-1] for step in historyFile]
val_mse_2d = [step['history_2d']['val_mse'][-1] for step in historyFile]
val_mae_2d = [step['history_2d']['val_mae'][-1] for step in historyFile]
val_acc_2d = [step['history_2d']['val_acc'][-1] for step in historyFile]
val_generalized_dice_coeff_2d = [step['history_2d']['val_generalized_dice_coeff'][-1] for step in historyFile]

time_3d = sum([step['unet_3d_time'] for step in historyFile], timedelta())
testDSC_3d = [step['testDSC_3d'] for step in historyFile]
loss_3d = [step['history_3d']['loss'][-1] for step in historyFile]
dice_coef_3d = [step['history_3d']['dice_coef'][-1] for step in historyFile]
mse_3d = [step['history_3d']['mse'][-1] for step in historyFile]
mae_3d = [step['history_3d']['mae'][-1] for step in historyFile]
acc_3d = [step['history_3d']['acc'][-1] for step in historyFile]
generalized_dice_coeff_3d = [step['history_3d']['generalized_dice_coeff'][-1] for step in historyFile]
val_loss_3d = [step['history_3d']['val_loss'][-1] for step in historyFile]
val_dice_coef_3d = [step['history_3d']['val_dice_coef'][-1] for step in historyFile]
val_mse_3d = [step['history_3d']['val_mse'][-1] for step in historyFile]
val_mae_3d = [step['history_3d']['val_mae'][-1] for step in historyFile]
val_acc_3d = [step['history_3d']['val_acc'][-1] for step in historyFile]
val_generalized_dice_coeff_3d = [step['history_3d']['val_generalized_dice_coeff'][-1] for step in historyFile]


f, axes = plt.subplots(1, 2)
# 2d
axes[0].plot(epochs, dice_coef_2d)
axes[0].plot(epochs, val_dice_coef_2d)
#axes[0].plot(epochs, testDSC_2d)
axes[0].legend(['Train', 'Valid', 'Test'])
axes[0].set_ylim(0, 1)
axes[0].set_title('UNet 2D')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('DSC')
axes[0].text(1,0,'time: ' + str(time_2d))

# 3d 
axes[1].plot(epochs, dice_coef_3d)
axes[1].plot(epochs, val_dice_coef_3d)
#axes[1].plot(epochs, testDSC_3d)
axes[1].legend(['Train', 'Valid', 'Test'])
axes[1].set_ylim(0, 1)
axes[1].set_title('UNet 3D')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('DSC')
axes[1].text(1,0,'time: ' + str(time_3d))

plt.show()
print("UNet 2D Train DSC: ", dice_coef_2d[-1])
print("UNet 2D Valid DSC: ", val_dice_coef_2d[-1])
print("UNet 2D Test DSC: ", testDSC_2d[-1])
print()
print("UNet 3D Train DSC: ", dice_coef_3d[-1])
print("UNet 3D Valid DSC: ", val_dice_coef_3d[-1])
print("UNet 3D Test DSC: ", testDSC_3d[-1])

