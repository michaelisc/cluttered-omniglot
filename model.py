import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys
import os
import time

slim = tf.contrib.slim

# from spatial_transformer import transformer



### Encoder ###

def encoder(images, feature_maps=16, dilated=False, reuse=False, scope='encoder'):
    with tf.variable_scope(scope, reuse=reuse):
        net = images
        end_points = OrderedDict()
        with slim.arg_scope([slim.conv2d],
                            padding='SAME', 
                            data_format='NHWC',
                            normalizer_fn=slim.layer_norm,
                            normalizer_params={'scale': False},
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG'), 
                            weights_regularizer=tf.contrib.layers.l2_regularizer(1e-9),
                            biases_initializer=None,
                            activation_fn=tf.nn.relu):
            
            net = slim.conv2d(net, num_outputs=feature_maps*(2**1), kernel_size=3, scope='encode1/conv3_1')
            end_points['encode1/conv3_1'] = net
            
            net = slim.avg_pool2d(net, [2, 2], scope='encode1/pool')
            net = slim.conv2d(net, num_outputs=feature_maps*(2**2), kernel_size=3, scope='encode2/conv3_1')
            end_points['encode2/conv3_1'] = net
            
            net = slim.avg_pool2d(net, [2, 2], scope='encode2/pool')
            net = slim.conv2d(net, num_outputs=feature_maps*(2**3), kernel_size=3, scope='encode3/conv3_1')
            end_points['encode3/conv3_1'] = net
            
            net = slim.avg_pool2d(net, [2, 2], scope='encode3/pool')
            net = slim.conv2d(net, num_outputs=feature_maps*(2**3), kernel_size=3, scope='encode4/conv3_1')
            end_points['encode4/conv3_1'] = net
            
            if dilated == False:
                net = slim.avg_pool2d(net, [2, 2], scope='encode4/pool')
            if dilated == False:
                net = slim.conv2d(net, num_outputs=feature_maps*(2**4), kernel_size=2, scope='encode5/conv3_1')
            elif dilated == True:
                net = slim.conv2d(net, num_outputs=feature_maps*(2**4), kernel_size=2, rate=2, scope='encode5/conv3_1')
            end_points['encode5/conv3_1'] = net
            
            if dilated == False:
                net = slim.avg_pool2d(net, [2, 2], scope='encode5/pool')
            net = slim.conv2d(net, num_outputs=feature_maps*(2**4), kernel_size=1, scope='encode6/conv3_1')
            end_points['encode6/conv3_1'] = net
            
    return net, end_points


### Decoder ###

#Decoder with skip connections        
def decoder(images, encoder_end_points, feature_maps=16, num_classes=2, reuse=False, scope='decoder'):
    def conv(fmaps, ks=3): return lambda net, name: slim.conv2d(net, num_outputs=fmaps, kernel_size=ks, scope=name)
    def skip(end_point): return lambda net, name: tf.concat([net, end_point], axis=3, name=name)
    unpool =  lambda net, name: tf.image.resize_nearest_neighbor(net, [2*tf.shape(net)[1], 2*tf.shape(net)[2]], name=name)

    layers = OrderedDict()
    layers['decode6/skip'] = skip(encoder_end_points['encode6/conv3_1'])
    layers['decode6/conv3_1'] = conv(feature_maps*(2**3), ks=1)
    #layers['decode6/unpool'] = unpool
    
    layers['decode5/skip'] = skip(encoder_end_points['encode5/conv3_1'])
    layers['decode5/conv3_1'] = conv(feature_maps*(2**3), ks=2)
    #layers['decode5/unpool'] = unpool
    
    layers['decode4/skip'] = skip(encoder_end_points['encode4/conv3_1'])
    layers['decode4/conv3_1'] = conv(feature_maps*(2**2))
    layers['decode4/unpool'] = unpool
    
    layers['decode3/skip'] = skip(encoder_end_points['encode3/conv3_1'])
    layers['decode3/conv3_1'] = conv(feature_maps*(2**2))
    layers['decode3/unpool'] = unpool
    
    layers['decode2/skip'] = skip(encoder_end_points['encode2/conv3_1'])
    layers['decode2/conv3_1'] = conv(feature_maps*(2**1))
    layers['decode2/unpool'] = unpool
    
    layers['decode1/skip'] = skip(encoder_end_points['encode1/conv3_1'])
    layers['decode1/classifier'] = lambda net, name: slim.conv2d(net, num_outputs=num_classes, kernel_size=3, activation_fn=None, scope=name)
    
    with tf.variable_scope(scope, reuse=reuse):
        net = images
        end_points = OrderedDict()
        with slim.arg_scope([slim.conv2d],
                            padding='SAME', 
                            normalizer_fn=slim.layer_norm,
                            normalizer_params={'scale': False},
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG'), 
                            weights_regularizer=tf.contrib.layers.l2_regularizer(1e-9), 
                            activation_fn=tf.nn.relu):
            for layer_name, layer_op in layers.items():
                net = layer_op(net, layer_name)
                end_points[layer_name] = net
            
    return net


### Siamese-U-Net ###

#Matching filter v8 and v9
def matching_filter(sample, target, mode='depthwise'):
    if mode == 'depthwise':
        conv2ims = lambda inputs : tf.nn.depthwise_conv2d(tf.expand_dims(inputs[0], 0),  # H,W,C -> 1,H,W,C
                                                      tf.expand_dims(inputs[1], 3),  # H,W,C -> H,W,C,1
                                                      strides=[1,1,1,1], padding="SAME") # Result of conv is 1,H,W,C
    else:
        conv2ims = lambda inputs : tf.nn.conv2d(tf.expand_dims(inputs[0], 0),  # H,W,C -> 1,H,W,C
                                                tf.expand_dims(inputs[1], 3),  # H,W,C -> H,W,C,1
                                                strides=[1,1,1,1], padding="SAME") # Result of conv is 1,H,W,1
    
    crosscorrelation = tf.map_fn(conv2ims, elems=[sample, target], dtype=tf.float32, name='crosscorrelation_1')
    
    return crosscorrelation[:, 0, :, :, :] # B,1,H,W,C -> B,H,W,C


def siamese_u_net(targets, images, feature_maps=24):
        
    #encode target
    targets_encoded, _ = encoder(targets, 
                              feature_maps=feature_maps, 
                              dilated=False,
                              reuse=False, 
                              scope='clean_encoder')


    images_encoded, images_encoded_end_points = encoder(images,  
                                                    feature_maps=feature_maps, 
                                                    dilated=True,
                                                    reuse=False, 
                                                    scope='clutter_encoder')


    #calculate crosscorrelation
    #target_encoded has to be [batch, 1, 1, fmaps] for this to work
    matched = matching_filter(images_encoded, targets_encoded, mode='standard')
    matched = matched * targets_encoded

    decoder_input = slim.layer_norm(matched, scale=False, center=False, scope='matching_normalization') 

    #get segmentation mask
    segmentations = decoder(decoder_input, 
                              images_encoded_end_points, 
                              feature_maps=feature_maps, 
                              reuse=False, 
                              scope='decoder')
    
    return segmentations

### Utils ###

def load_dataset_train(dataset_dir):
    
    #load training data
    path = dataset_dir + 'train/'
    ims_train = np.load(path + 'images.npy', mmap_mode='r')
    seg_train = np.load(path + 'segmentation.npy', mmap_mode='r')
    tar_train = np.load(path + 'targets.npy', mmap_mode='r')
    
    return ims_train, seg_train, tar_train

def load_dataset_val(dataset_dir, subset, old_notation=False):
    
    if subset == 'train':
        #load val_train data

        path = dataset_dir + 'val-train/'
        ims_val = np.load(path + 'images.npy')
        seg_val = np.load(path + 'segmentation.npy')
        tar_val = np.load(path + 'targets.npy')
        
    elif subset == 'eval':
        #load val_eval data

        path = dataset_dir + 'val-one-shot/'
        ims_val = np.load(path + 'images.npy')
        seg_val = np.load(path + 'segmentation.npy')
        tar_val = np.load(path + 'targets.npy') 
        
    else:
        print(subset + ' is not a valid subset')
    
    return ims_val, seg_val, tar_val

def make_batch(batch_size, ims, seg, tar, perms=np.zeros(10), step=0):
    images_batch = np.zeros((batch_size,ims.shape[1],ims.shape[2],3))
    labels_batch = np.zeros((batch_size,ims.shape[1],ims.shape[2],1))
    target_batch = np.zeros((batch_size,tar.shape[1],tar.shape[2],3))
    
    if all(perms == 0):
        perms = np.random.permutation(tar.shape[0])
        
    ntarget_index = np.random.randint(tar.shape[1])
        
    for i in range(batch_size):
        index = perms[step*batch_size+i]
        images_batch[i,:,:,:] = ims[index,:,:,:]
        labels_batch[i,:,:,:] = seg[index,:,:,:]
        target_batch[i,:,:,:] = tar[index,:,:,:]
    
    return images_batch, target_batch, labels_batch

#IoU claculation routine
def calculate_IoU(segmentations, labels, threshold=0.3):
    pred = tf.squeeze(labels, axis=-1)
    seg_softmax = tf.nn.softmax(segmentations, axis=-1)
    seg = tf.cast(seg_softmax[...,1] > threshold, tf.int32)
    IoU = tf.reduce_sum(pred*seg, axis=(1,2))/(tf.reduce_sum(pred, axis=(1,2))+tf.reduce_sum(seg, axis=(1,2))-tf.reduce_sum(pred*seg, axis=(1,2)))
    clean_IoU = tf.where(tf.is_nan(IoU), tf.ones_like(IoU), IoU) #Remove NaNs which appear when the target does not exist
    
    return clean_IoU


### Training ###

def training(dataset_dir, 
             logdir, 
             epochs, 
             model='siamese-u-net',
             feature_maps=24,
             batch_size=250,
             learning_rate=0.0005, 
             initial_training=True, 
             maximum_number_of_steps=0):
    
    # Currently only the siamese-u-net is implemented
    assert model in ['siamese-u-net', 'mask_net']
    
    with tf.Graph().as_default():
        
        #Define logging parameters
        t = time.time()
        OSEG_CKPT_FILE = logdir + 'Run.ckpt'
        weight_logging = True
        if not tf.gfile.Exists(logdir):
            tf.gfile.MakeDirs(logdir)
        
        #Load dataset
        print('Loading dataset: ' + dataset_dir)
        ims_train, seg_train, tar_train = load_dataset_train(dataset_dir)
        ims_val_train, seg_val_train, tar_val_train = load_dataset_val(dataset_dir, subset='train')
        ims_val_eval, seg_val_eval, tar_val_eval = load_dataset_val(dataset_dir, subset='eval')
        print('Done loading dataset')
        
        #Define training parameters
        batch_size = batch_size
        max_steps = tar_train.shape[0]//batch_size
        if maximum_number_of_steps != 0:
            print('Going to run for %.d steps'%(np.min([epochs*max_steps, maximum_number_of_steps])))
        else:
            print('Going to run for %.d steps'%(epochs*max_steps))
        
        #Get dataset information and statistics
        szx = tar_train.shape[1]
        szy = tar_train.shape[2]
        nway = ims_train.shape[0]
        if ims_train.shape[0] >= 1000:
            mean = np.mean(ims_train[:1000,...])
            std = np.std(ims_train[:1000,...])
        else:
            mean = np.mean(ims_train)
            std = np.std(ims_train)      

        #generate tensorflow placeholders and variables
        images = tf.placeholder(tf.float32, shape=[batch_size,ims_train.shape[1],ims_train.shape[2],3], name='images')
        labels = tf.placeholder(tf.int32, shape=[batch_size,ims_train.shape[1],ims_train.shape[2],1], name='labels')
        targets = tf.placeholder(tf.float32, shape=[batch_size,tar_train.shape[1],tar_train.shape[2],3], name='targets')
        learn_rate = tf.Variable(learning_rate)
        
        #preprocess images
        targets = (targets - mean)/std
        images = (images - mean)/std
        
        #get predictions
        if model == 'siamese-u-net':
            segmentations = siamese_u_net(targets, images, feature_maps=feature_maps)
        elif model == 'mask_net':
            segmentations = mask_net(targets, images, feature_maps=feature_maps)
        
        #Update batch norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        #Define Losses
        seg_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=segmentations, scope='losses')
        reg_loss = tf.add_n(tf.losses.get_regularization_losses())
        loss = seg_loss + reg_loss

        #Training step
        optimizer = tf.train.AdamOptimizer(learn_rate)
        global_step = tf.Variable(0, name = 'global_step', trainable = False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        #Calculate metrics: accuracy
        mean_IoU = tf.reduce_mean(calculate_IoU(segmentations, labels))

        #Create summaries
        tf.summary.scalar('segmentation_loss', seg_loss)
        tf.summary.scalar('regularization_loss', reg_loss)
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('mean_IoU', mean_IoU)
        
        #Collect summaries
        summary = tf.summary.merge_all()

        #Logging
        saver = tf.train.Saver()
        restorer = tf.train.Saver(slim.get_model_variables())

        
        #Start Session
        with tf.Session() as sess:
            #Initialize logging files
            summary_writer_train = tf.summary.FileWriter(logdir, sess.graph)
            summary_writer_val_train = tf.summary.FileWriter(logdir + 'val_train')
            summary_writer_val_eval = tf.summary.FileWriter(logdir + 'val_eval')
            
            #Initialize from scratch or finetune from previous training
            sess.run(tf.global_variables_initializer())
            if initial_training == True:
                step_count = 0
            else:
                step_count = (step_count//100) * 100
                restorer.restore(sess, OSEG_CKPT_FILE)

            #Training loop
            print('Starting to train')
            duration = []
            for epoch in range(epochs):
                #Learning rate schelude
                if epoch == epochs//2 or epoch == 3*epochs//4 or epoch == 7*epochs//8:
                    learning_rate = learning_rate/2
                    print('lowering learning rate to %.4f'%(learning_rate))
                #Shuffle samples
                perms = np.random.permutation(tar_train.shape[0])
        
                #Run trainings step
                for step in range(max_steps):
                    start_time = time.time()
                    
                    images_batch, target_batch, labels_batch = make_batch(batch_size, ims_train, seg_train, tar_train, perms=perms, step=step)
                    _, loss_value = sess.run([train_op, loss],
                                             feed_dict = {targets: target_batch,
                                                          images: images_batch,
                                                          labels: labels_batch,
                                                          learn_rate: learning_rate})
                    step_count = step_count + 1
                    duration.append(time.time()-start_time)

                    
                    #Evaluate 
                    if step_count % 100 == 0 or step_count == 1:
                        #Evaluate and print training error and IoU
                        summary_str_train, tf_IoU = sess.run([summary, mean_IoU], 
                                                             feed_dict = {targets: target_batch, 
                                                                          images: images_batch, 
                                                                          labels: labels_batch})
                        summary_writer_train.add_summary(summary_str_train, step_count)
                        summary_writer_train.flush()
                        print('Step %5d: loss = %.4f mIoU: %.3f (%.3f sec)' 
                              % (step_count, np.mean(loss_value), tf_IoU, np.mean(duration)))
                        duration = []
                        
                        #evaluate val_train
                        images_batch, target_batch, labels_batch = make_batch(batch_size, ims_val_train, seg_val_train, tar_val_train)
                        summary_str = sess.run(summary, feed_dict={targets: target_batch, 
                                                                       images: images_batch, 
                                                                       labels: labels_batch})
                        summary_writer_val_train.add_summary(summary_str, step_count)
                        summary_writer_val_train.flush()
                        
                        #evaluate val_eval
                        images_batch, target_batch, labels_batch = make_batch(batch_size, ims_val_eval, seg_val_eval, tar_val_eval)
                        summary_str = sess.run(summary, feed_dict={targets: target_batch, 
                                                                       images: images_batch, 
                                                                       labels: labels_batch})
                        
                        summary_writer_val_eval.add_summary(summary_str, step_count)
                        summary_writer_val_eval.flush()

                        
                    #Create checkpoint    
                    if step_count % 100 == 0 or step_count == epochs*max_steps:
                        checkpoint_file = os.path.join(logdir, 'Run.ckpt')
                        saver.save(sess, checkpoint_file)
                        
                    if step_count == maximum_number_of_steps:
                        return
                        

        print('Total time: ', time.time()-t)
        

        
### Evaluate ###

def _center_of_mass(image):
    
    #returns the pixel corresponding to the center of mass of the segmentation mask
    #if no pixel is segmented [-1,-1] is returned 
    
    sz = image.get_shape().as_list()
    batch_size = sz[0]
    szx = sz[1]
    szy = sz[2]
    
    e = 0.00001
    
    x,y = tf.meshgrid(list(range(0,szx)),list(range(0,szy)))
    x = tf.cast(tf.tile(tf.expand_dims(tf.expand_dims(x, axis=-1), axis=0), [batch_size, 1, 1, 1]), tf.float32)
    y = tf.cast(tf.tile(tf.expand_dims(tf.expand_dims(y, axis=-1), axis=0), [batch_size, 1, 1, 1]), tf.float32)
    comx = (tf.reduce_sum(x * image, axis=[1,2,3])+e)//(tf.reduce_sum(image, axis=[1,2,3])-e)
    comy = (tf.reduce_sum(y * image, axis=[1,2,3])+e)//(tf.reduce_sum(image, axis=[1,2,3])-e)
    
    return comx, comy

def _get_shift_xy(image):
    comx, comy = _center_of_mass(image)
    sz = image.get_shape().as_list()
    szx = sz[1]
    szy = sz[2]
    shiftx = 2*comx/(szx-1) - 1
    shifty = 2*comy/(szy-1) - 1
    shift = tf.stack([shiftx, shifty])
    
    return shift

#Network training
def evaluation(dataset_dir, 
               logdir,
               model='siamese-u-net',
               feature_maps=24,
               batch_size=250,
               threshold=0.3):
    
    # Currently only the siamese-u-net is implemented
    assert model in ['siamese-u-net', 'mask_net']
    
    with tf.Graph().as_default():
        
        #Define logging parameters
        OSEG_CKPT_FILE = logdir + 'Run.ckpt'
        
        #Load dataset
        print('Loading dataset: ' + dataset_dir)
        ims_val_train, seg_val_train, tar_val_train = load_dataset_val(dataset_dir, subset='train')
        ims_val_eval, seg_val_eval, tar_val_eval = load_dataset_val(dataset_dir, subset='eval')
        print('Done loading dataset')
        
        #Define training parameters
        batch_size = batch_size
        max_steps = tar_val_train.shape[0]//batch_size
        
        #Get dataset information and statistics
        szx = tar_val_train.shape[1]
        szy = tar_val_train.shape[2]
        nway = ims_val_train.shape[0]
        if ims_val_train.shape[0] >= 1000:
            mean = np.mean(ims_val_train[:1000,...])
            std = np.std(ims_val_train[:1000,...])
        else:
            mean = np.mean(ims_val_train)
            std = np.std(ims_val_train)      

        #generate tensorflow placeholders and variables
        images = tf.placeholder(tf.float32, shape=[batch_size,ims_val_train.shape[1],ims_val_train.shape[2],3], name='images')
        labels = tf.placeholder(tf.int32, shape=[batch_size,ims_val_train.shape[1],ims_val_train.shape[2],1], name='labels')
        targets = tf.placeholder(tf.float32, shape=[batch_size,tar_val_train.shape[1],tar_val_train.shape[1],3], name='targets')
        
        #preprocess images
        targets = (targets - mean)/std
        images = (images - mean)/std
        
        #get predictions
        if model == 'siamese-u-net':
            segmentations = siamese_u_net(targets, images, feature_maps=feature_maps)
        elif model == 'mask_net':
            segmentations = mask_net(targets, images, feature_maps=feature_maps)
        

        #Calculate metrics: IoU
        mean_IoU = calculate_IoU(segmentations, labels, threshold=threshold)
        #Calculate metrics: Localization Accuracy
        seg_softmax = tf.nn.softmax(segmentations, axis=-1)
        seg = tf.expand_dims(tf.cast(seg_softmax[...,1] > threshold, tf.float32), axis=-1)
        lcomx, lcomy = _center_of_mass(tf.cast(labels, tf.float32))
        comx, comy = _center_of_mass(seg)
        euclidian_distance = tf.sqrt((lcomx-comx)**2 + (lcomy-comy)**2)
        distance_metric = tf.cast(euclidian_distance < 5, tf.float32)
        mean_dist = tf.reduce_mean(distance_metric)

        #Logging
        saver = tf.train.Saver()
        restorer = tf.train.Saver(slim.get_model_variables())

        
        #Start Session
        with tf.Session() as sess:

            #Initialize from scratch or finetune from previous training
            sess.run(tf.global_variables_initializer())
            restorer.restore(sess, OSEG_CKPT_FILE)

            #Training loop
            perms = np.random.permutation(tar_val_train.shape[0])

            #Run trainings step
            val_IoU = [0 for x in range(max_steps)]
            os_IoU = [0 for x in range(max_steps)]
            val_distances = [0 for x in range(max_steps)]
            os_distances = [0 for x in range(max_steps)]
            for step in range(max_steps):

                images_batch, target_batch, labels_batch = make_batch(batch_size, ims_val_train, seg_val_train, tar_val_train, perms=perms, step=step)
                val_IoU[step], val_distances[step] = sess.run([mean_IoU, mean_dist],
                                         feed_dict = {targets: target_batch,
                                                      images: images_batch,
                                                      labels: labels_batch})
                
                images_batch, target_batch, labels_batch = make_batch(batch_size, ims_val_eval, seg_val_eval, tar_val_eval, perms=perms, step=step)
                os_IoU[step], os_distances[step] = sess.run([mean_IoU, mean_dist],
                                         feed_dict = {targets: target_batch,
                                                      images: images_batch,
                                                      labels: labels_batch})
                
            print('Valiadation IoU: %.3f'%(np.mean(val_IoU)), 'Validation Distance: %.3f'%(np.mean(val_distances)))
            print('One-Shot IoU: %.3f'%(np.mean(os_IoU)), 'One-Shot Distance: %.3f'%(np.mean(os_distances)))