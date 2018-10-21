from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt

pixelcnn = EventAccumulator('logs/pixelcnn')
noncausal = EventAccumulator('logs/noncausal')
denoising = EventAccumulator('logs/denoising')
pixelcnn.Reload()
noncausal.Reload()
denoising.Reload()

_, step_nums, vals = zip(*pixelcnn.Scalars('train_loss'))
pixelcnn_train = (step_nums, vals)

_, step_nums, vals = zip(*pixelcnn.Scalars('test_loss'))
pixelcnn_test = (step_nums, vals)

_, step_nums, vals = zip(*noncausal.Scalars('train_loss'))
noncausal_train = (step_nums, vals)

_, step_nums, vals = zip(*noncausal.Scalars('test_loss'))
noncausal_test = (step_nums, vals)

_, step_nums, vals = zip(*denoising.Scalars('train_loss'))
denoising_train = (step_nums, vals)

_, step_nums, vals = zip(*denoising.Scalars('test_loss'))
denoising_test = (step_nums, vals)

plt.plot(*pixelcnn_train, label='PixelCNN Train')
plt.plot(*pixelcnn_test, label='PixelCNN Test')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.xticks([0,100000,200000])
plt.legend()
plt.show()
plt.close()
plt.plot(*noncausal_train, label='Noncausal Train')
plt.plot(*noncausal_test, label='Noncausal Test')
plt.plot(*denoising_train, label='Denoising Train')
plt.plot(*denoising_test, label='Denoising Test')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.xticks([0,100000,200000])
plt.legend()
plt.show()

