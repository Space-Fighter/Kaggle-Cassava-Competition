# from sklearn.model_selection import KFold

# oof_pred = []; oof_labels = []; history_list = []

# kfold = KFold(FOLDS, shuffle = True, random_state = SEED)

# for f, (train_idx, val_idx) in enumerate(kfold.split(TRAINING_FILENAMES)):
#     print('-'*9)
#     print(f"|FOLD: {f+1}|")
#     print('-' * 9)

#     #show fold info
#     if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)
#     K.clear_session()
    
#     # Gret the training and validation filenames
#     TRAIN_FN = tf.io.gfile.glob([GCS_PATH + '/*.tfrec' % x for x in train_idx])    
#     VALID_FN = tf.io.gfile.glob([GCS_PATH + '/*.tfrec' % x for x in val_idx])
#     ct_train = count_data_items(TRAIN_FN)
#     ct_valid = count_data_items(VALID_FN)
    
#     print('Training Data Count: ', ct_train)
#     print('Validation Data Count: ', ct_valid)
#     step_size = (ct_train // BATCH_SIZE)
#     valid_step_size = (ct_valid // BATCH_SIZE)
#     total_steps=(total_epochs * step_size)
#     warmup_steps=(warmup_epochs * step_size)
#     # Loading in our data
#     train_ds = strategy.experimental_distribute_dataset(get_dataset(TRAIN_FN, labeled=True, grid_mask=False, all_aug=True, one_hot=False, 
#                                                         cutmixup=True, course_drop=False, shuffle=True, repeat=True))

    
#     valid_ds = strategy.experimental_distribute_dataset(get_dataset(VALID_FN,course_drop=False, grid_mask=False, all_aug=False, 
#                          cutmixup=False, one_hot=True, labeled=True, return_image_names=False, repeat=False, shuffle=False))
    
    
#     train_data_iter = iter(train_ds)
#     valid_data_iter = iter(valid_ds)
    
    
#     # Step functions
#     @tf.function
#     def train_step(data_iter):
#         def train_step_fn(x, y):
#             with tf.GradientTape() as tape:
#                 probabilities = model(x, training=True)
#                 loss = loss_fn(y, probabilities, label_smoothing=.3)
#             gradients = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#             # update metrics
#             train_accuracy.update_state(y, probabilities)
#             train_loss.update_state(loss)
#         for _ in tf.range(step_size):
#             strategy.experimental_run_v2(train_step_fn, next(data_iter))

#     @tf.function
#     def valid_step(data_iter):
#         def valid_step_fn(x, y):
#             probabilities = model(x, training=False)
#             loss = loss_fn(y, probabilities)
#             # update metrics
#             valid_accuracy.update_state(y, probabilities)
#             valid_loss.update_state(loss)
#         for _ in tf.range(valid_step_size):
#             strategy.experimental_run_v2(valid_step_fn, next(data_iter))
    
    
#     # Model
#     model_path = f'model_{fold}.h5'
#     with strategy.scope():
#         model = model_fn((None, None, CHANNELS), N_CLASSES)
#         unfreeze_model(model) # unfreeze all layers except "batch normalization"
        
#         optimizer = optimizers.Adam(learning_rate=lambda: lrfn(tf.cast(optimizer.iterations, tf.float32)))
#         loss_fn = losses.categorical_crossentropy

#         train_accuracy = metrics.CategoricalAccuracy()
#         valid_accuracy = metrics.CategoricalAccuracy()
#         train_loss = metrics.Sum()
#         valid_loss = metrics.Sum()
    
    
#     # Setup training loop
#     step = 0
#     epoch_steps = 0
#     patience_cnt = 0
#     best_val = 0
#     history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

#     ### Train model
#     for epoch in range(EPOCHS):
#         epoch_start_time = time.time()

#         # Run training step
#         train_step(train_data_iter)
#         epoch_steps += step_size
#         step += step_size


#         # Validation run at the end of each epoch
#         if (step // step_size) > epoch:
#             # Validation run
#             valid_epoch_steps = 0
#             valid_step(valid_data_iter)
#             valid_epoch_steps += valid_step_size

#             # Compute metrics
#             history['accuracy'].append(train_accuracy.result().numpy())
#             history['loss'].append(train_loss.result().numpy() / (BATCH_SIZE * epoch_steps))
#             history['val_accuracy'].append(valid_accuracy.result().numpy())
#             history['val_loss'].append(valid_loss.result().numpy() / (BATCH_SIZE * valid_epoch_steps))

#             # Report metrics
#             epoch_time = time.time() - epoch_start_time
#             print(f'\nEPOCH {epoch+1}/{EPOCHS}')
#             print(f'time: {epoch_time:0.1f}s',
#                   f"loss: {history['loss'][-1]:0.4f}",
#                   f"accuracy: {history['accuracy'][-1]:0.4f}",
#                   f"val_loss: {history['val_loss'][-1]:0.4f}",
#                   f"val_accuracy: {history['val_accuracy'][-1]:0.4f}",
#                   f'lr: {lrfn(tf.cast(optimizer.iterations, tf.int32).numpy()):0.4g}')

#             # Early stopping monitor
#             if history['val_accuracy'][-1] >= best_val:
#                 best_val = history['val_accuracy'][-1]
#                 model.save_weights(model_path)
#                 print(f'Saved model weights at "{model_path}"')
#                 patience_cnt = 1
#             else:
#                 patience_cnt += 1
#             if patience_cnt > ES_PATIENCE:
#                 print(f'Epoch {epoch:05d}: early stopping')
#                 break

                
#             # Set up next epoch
#             epoch = step // step_size
#             epoch_steps = 0
#             train_accuracy.reset_states()
#             train_loss.reset_states()
#             valid_accuracy.reset_states()
#             valid_loss.reset_states()
    
    
#     ### RESULTS
#     print(f"#### FOLD {fold+1} OOF Accuracy = {np.max(history['val_accuracy']):.3f}")
    
#     history_list.append(history)
#     # Load best model weights
#     model.load_weights(model_path)

#     # OOF predictions
#     ds_valid = get_dataset(VALID_FILENAMES, ordered=True)
#     oof_labels.append([target.numpy() for img, target in iter(ds_valid.unbatch())])
#     x_oof = ds_valid.map(lambda image, target: image)
#     oof_pred.append(np.argmax(model.predict(x_oof), axis=-1))
#     print('='*126)
