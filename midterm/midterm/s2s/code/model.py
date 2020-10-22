#coding=utf-8
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.ops import variable_scope as vs    
import random
import numpy as np
import copy


class Seq2Seq():
    def __init__(self,emb_size,hid_size,vocab_size,y_seq_len):
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.vocab_size = vocab_size
        self.embedding = tf.get_variable('word_embedding', [self.vocab_size, self.emb_size])
        self.y_seq_len = y_seq_len
        self.learning_rate = 1e-3
    def _add_placeholder(self):
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.decoder_inputs = tf.placeholder(tf.int32, [None, self.y_seq_len], name='decoder_inputs')
        self.decoder_inputs_2 = tf.placeholder(tf.int32, [None], name='decoder_inputs2')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
    def _add_embedding_look_up(self):

        self.encoder_inputs_embed = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)

        self.decoder_targets_embeddded = [tf.nn.embedding_lookup(self.embedding, x) for x in tf.unstack(self.decoder_inputs, axis=1)]
    
    def length(self, data):

        used = tf.sign(data) # B x 100
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)
        return used, length
   
    def _add_encoder(self, encoder_inputs, scope_name='Encoder'):

        encoder_mask, seq_len = self.length(self.encoder_inputs)
        self.seq_len = seq_len


        with tf.variable_scope(scope_name) as scope:
            
            cell_fn = tf.nn.rnn_cell.GRUCell
            
            cell_fw = cell_fn(self.hid_size)
            cell_bw = cell_fn(self.hid_size)
            

            ((encoder_fw_outputs, encoder_bw_outputs),
            (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_bw, cell_fw,
                                                                                           encoder_inputs,
                                                                                           dtype=tf.float32,
                                                                                           sequence_length=seq_len,
                                                                                           swap_memory=True)
            encoder_outputs = tf.concat([encoder_fw_outputs, encoder_bw_outputs], 2)
            #encoder_state = get_BiLastState(encoder_fw_state, encoder_bw_state, self.rnn_type)

        return encoder_outputs,encoder_mask,tf.concat([encoder_fw_state, encoder_bw_state],1) # 2d

    def _add_decoder(self,decoder_input, dec_state, source_outputs,decoder_cell,mode='decode'):

            
        initial_state_attention = (mode == 'decode')
        outputs, out_state = tf.contrib.legacy_seq2seq.attention_decoder(decoder_input, dec_state, source_outputs,decoder_cell,initial_state_attention = initial_state_attention,scope='decoder')

        return outputs, out_state
    def _train_op(self):

        self.params = tf.trainable_variables()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.gradients = tf.gradients(self._loss, self.params)
        self.clipped_gradients, norm = tf.clip_by_global_norm(self.gradients, self.max_gradient_norm)
        self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.params))
    def compute_loss(self,logits,labels,mask):
        probs = tf.nn.softmax(logits,-1)
        labels = tf.one_hot(labels,probs.shape[-1].value) # B x 100 x v
        loss = - tf.reduce_sum(tf.log(probs+1e-12)*labels,-1) #B x 100
        loss = tf.reduce_mean(loss * tf.cast(mask,tf.float32),-1) 
        return loss
    def build_model(self):
        self._add_placeholder()
        self._add_embedding_look_up()

        self.source_outputs, self.source_mask,self.source_state = self._add_encoder(self.encoder_inputs_embed)
        
        decoder_cell = tf.contrib.rnn.GRUCell(self.hid_size*2,name="de")
        # Train
        with tf.variable_scope('decode'):
            de_output, out_state = self._add_decoder(self.decoder_targets_embeddded,self.source_state,self.source_outputs,decoder_cell,mode="train")

            de_output = tf.stack(de_output,1)
            softmax_w = tf.get_variable('softmax_w', [self.hid_size*2, self.vocab_size])
            softmax_b = tf.get_variable('softmax_b', [self.vocab_size])
            logits = tf.einsum('ijk,kl->ijl',de_output,softmax_w) + softmax_b
            target_mask,_ = self.length(self.targets)
            loss = self.compute_loss(logits,self.targets,target_mask)
            self.loss = tf.reduce_mean(loss)
            

        # Inference
        with tf.variable_scope('decode'):
            vs.get_variable_scope().reuse_variables()
            decoder_in = [tf.nn.embedding_lookup(self.embedding,self.decoder_inputs_2)]
            decoder_state = self.source_state
            output = []
            for i in range(self.y_seq_len):
                de_output, decoder_state = self._add_decoder(decoder_in,decoder_state,self.source_outputs,decoder_cell,mode="decode")    
                de_output = de_output[0]
                logits = tf.matmul(de_output, softmax_w) + softmax_b #B x V           
                index = tf.argmax(de_output,-1) # B
                decoder_in =  [tf.nn.embedding_lookup(self.embedding, index)] 
                output.append(index) #[B] * 100
        self.output = tf.stack(output,1) # B x 100

        #optimizer
        self.params = tf.trainable_variables()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.gradients = tf.gradients(self.loss, self.params)
        self.clipped_gradients, norm = tf.clip_by_global_norm(self.gradients, 10.0)
        self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.params))
        self.saver = tf.train.Saver(self.params,max_to_keep=1)                
    def train(self,sess,inputs,outputs):
        
        decoder_in = outputs[:,:-1]
        targets = outputs[:,1:]
        feed_dict = {
                    self.encoder_inputs: inputs,
                    self.decoder_inputs:decoder_in,
                    self.targets:targets}
        loss,_ = sess.run([self.loss,self.train_op],feed_dict)
        return loss
    
    def eval(self,sess,inputs,decoder_in):
        feed_dict = {
                    self.encoder_inputs: inputs,
                    self.decoder_inputs_2:decoder_in}
        output = sess.run(self.output,feed_dict)
        return output
