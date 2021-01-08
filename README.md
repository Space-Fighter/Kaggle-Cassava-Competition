# Kaggle-Cassava-Competition

What i have tried - 

1. V0.1 - Simple efficientnet with not any special algorithm with image data generator. Didn't turn out well.
    a. Submission without TTA performed very badly
    b. Submission with TTA peformed better but no significant improvement
2. Tried using Keras Tuner. Was giving me lots of errors **on TPU.**
    Future plans for this - Train the tuner on GPU
3. V0.2 - 🚧👷Work In Progress👷🚧
    Using Kfold cross validation
    Using a nice learning rate scheduler
    ```py
    def get_lr_callback(batch_size=8, show = False):
      lr_start   = 0.000015000
      lr_max     = 0.000000250 * strategy.num_replicas_in_sync * batch_size
      lr_min     = 0.000001
      lr_ramp_ep = 5
      lr_sus_ep  = 0
      lr_decay   = 0.8

      def lrfn(epoch):
          if epoch < lr_ramp_ep:
              lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

          elif epoch < lr_ramp_ep + lr_sus_ep:
              lr = lr_max

          else:
              lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

          return lr
      if show:
          plt.figure(figsize = (8, 5))
          plt.plot(np.arange(1, 12), [lrfn(x) for x in np.arange(1, 12)], marker = 'o')
          plt.xlabel('epoch')
          plt.ylabel('learaning_rate');
          plt.show()

      lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
      return lr_callback

    get_lr_callback(batch_size=BATCH_SIZE, show = True)
```
    Taken from here - [Notebook](https://www.kaggle.com/awsaf49/efficientnetb6-512-cutmixupdropout-tpu-train/notebook)
Future Plans    
4. Use Cutmix dropout
5. Use cosine decay
6. Fit the batchnorm of efficientnet to the data
7. Try getting in resnext
