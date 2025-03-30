import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util.encoder import Encoder, EncoderBlock
from util.embedding import EmbeddingLayer, PositionalEncoding
from util.decoder import Decoder, ProjectionLayer, DecoderBlock
from util.attention import MultiHeadAttention
from util.feedback import FeedForward
from util.transformer import Transformer
from util.dataset_encoder import casual_mask
from util.log import LOG

class ModelBuilder():
    def __init__(self):
        pass

    def build_model(self, source_vocab_size: int, target_vocab_size: int, 
                    source_seq_len: int, target_seq_len: int, 
                    d_model: int=512, num_blocks: int=6,
                    num_heads: int=8, dropout_rate: float=0.1,
                    d_ff: int=2048) -> Transformer:
        
        # create embedding layers
        source_embed = EmbeddingLayer(d_model, source_vocab_size)
        target_embed = EmbeddingLayer(d_model, target_vocab_size)

        # create the positional encoding layers
        source_pos = PositionalEncoding(d_model, source_seq_len, dropout_rate)
        target_pos = PositionalEncoding(d_model, target_seq_len, dropout_rate)

        # create the encoder-block list
        encoderblocklist = []
        for _ in range(num_blocks):
            multihead_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
            feed_forward = FeedForward(d_model, d_ff, dropout_rate)
            encoder_block = EncoderBlock(multihead_attention, feed_forward, dropout_rate)
            encoderblocklist.append(encoder_block)
        
        # create the encoder
        encoder = Encoder(nn.ModuleList(encoderblocklist))
        
        # create the decoder-block list
        decoderblocklist = []
        for _ in range(num_blocks):
            masked_multihead_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
            cross_multihead_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
            feed_forward = FeedForward(d_model, d_ff, dropout_rate)
            decoder_block = DecoderBlock(masked_multihead_attention, cross_multihead_attention, feed_forward, dropout_rate)
            decoderblocklist.append(decoder_block)
        
        # create the decoder
        decoder = Decoder(nn.ModuleList(decoderblocklist))

        # create the projection layer
        projection_layer = ProjectionLayer(d_model, target_vocab_size)

        # initiate the model
        model = Transformer(encoder, decoder, source_embed, target_embed, source_pos, target_pos, projection_layer)

        # intitialize the model parameters using xavier uniform method. Once
        # training begins, the parameters will be updated by the network
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return model
    
    def run_validation(self, model, validation_ds, tokenizer_en, tokenizer_target, max_seq_len, device, print_msg, global_step):
        model.eval()
        count = 0

        with torch.no_grad():
            for batch in validation_ds:
                count += 1
                encoder_input = batch['encoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)

                cls_id = tokenizer_target.token_to_id('[CLS]')
                sep_id = tokenizer_target.token_to_id('[SEP]')

                # computing the output of the encoder for the source sequence
                encoder_output = model.encode(encoder_input, encoder_mask)

                # for prediction task, the first token that goes in decoder input is the [CLS]
                decoder_input = torch.empty(1, 1).fill_(cls_id).type_as(encoder_input).to(device)

                # keep adding the output back to the input until the [SEP] - end token is receieved
                while True:
                    # check if max length is recieved
                    if decoder_input.size(1) == max_seq_len:
                        break
                    
                    # recreate mask each time th enew output is added the decoder input for next token prediction
                    decoder_mask = casual_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

                    # apply projection only to next token
                    out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

                    # apply projection onlyt to the next token
                    prob = model.project(out[:, -1])

                    # select the token with the highest probability which is a greedy search implementation
                    _, next_word = torch.max(prob, dim=1)
                    decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1)

                    # check if new token is the end of token
                    if next_word == sep_id:
                        break
                
                # final output is the concatinated decoder input till the end token is reached
                model_out = decoder_input.squeeze(0)

                source_text = batch['source_text'][0]
                target_text = batch['target_text'][0]
                model_out_text = tokenizer_target(model_out.detach().cpu().numpy())

                # print the source, target, and model output
                print_msg('-' * 55)
                print_msg(f'Source Text: {source_text}')
                print_msg(f'Target Text: {target_text}')
                print_msg(f'Predicted by Gater: {model_out_text}')
            
                if count == 2:
                    break

    def train_model(self, model, device, tokenizer_en, tokenizer_target, max_seq_len,
                    train_dataloader: DataLoader, val_dataloader: DataLoader, 
                    preloaded_epoch=None):
        # validation cycle will run for 20 ephochs
        EPOCHS = 10
        init_epoch = 0
        global_step = 0

        # using Adam optimization algo to hold current state and update 
        # the parameters based on computer gradients
        optimizer = torch.optim.Adam(model.paramters(), lr=1e-4, eps=1e-9)

        # if preload_epoch is not none, that means the training will 
        # start with the weights, optimizer that has been last saved 
        # and start with proladed epoch + 1
        if preloaded_epoch is not None:
            model_filename = f'./gater/model_{preloaded_epoch}.pt'
            state = torch.loard(model_filename)
            model.load_state_dict(state['model_state_dict'])
            inital_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['opimizer_state_dict'])
            global_step = state['gloabl_step']

        # CrossEntropyLoss computes the difference between the 
        # difference between the projection output and target label
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_en.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

        for epoch in range(init_epoch, EPOCHS):
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f'Processing Epoch {epoch:02d}')

            for batch in batch_iterator:
                encoder_input = batch['encoder_input'].to(device)   # (b, seq_len)
                decoder_input = batch['decoder_input'].to(device)   # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device)     # (B, 1, 1 seq_len)
                decoder_mask = batch['decoder_mask'].to(device)     # (B, 1, seq_len, seq_len)
                target_label = batch['target_label'].to(device)     # (B, seq_len)
            
            # run tensors through the encoder, decoder, and projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)      # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask,
                                            decoder_input, decoder_mask)     # (B, seq_len, d_model)
            projection_output = model.project(decoder_output)               # (B, seq_len)

            # compute loss using cross entropy
            loss = loss_fn(projection_output.view(-1, tokenizer_target.get_vocob_size()), target_label.view(-1))
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

            # backpropgate the loss
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zer_grad(set_to_none=True)

            global_step += 1

        # :: VALIDATION BLOCK HERE ::
        # runs every epoch after training block is complete
        self.run_valid(model, val_dataloader, tokenizer_en, tokenizer_target, 
                       max_seq_len, device, lambda msg: batch_iterator.write(msg), 
                       global_step)
        
        # save model at end of evey epoch
        model_filename = f'./gater/model_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    # train_model()


# build model, test, and show model architecture
MAX_SEQ_LENGTH = 155
build = ModelBuilder()
model = build.build_model(tokenizer_en.get_vocab_size(), tokenizer_target.get_vocab_size(), max_seq_len, d_model_512).to(device)
LOG.info(model)