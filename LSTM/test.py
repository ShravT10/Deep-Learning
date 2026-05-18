import tensorflow as tf 
import os 
import joblib
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



model = tf.keras.models.load_model('LSTM/LSTM.keras')
tokenizer = joblib.load('LSTM/tokenizer.joblib')

def sentiment_analysis(sentence):

    embed = tokenizer.texts_to_sequences(sentence)

    embed_pad = tf.keras.preprocessing.sequence.pad_sequences(
    embed,
    maxlen = 200,
    padding = 'post',
    truncating = 'post'
    )

    score = model.predict(embed_pad)[0][0]

    return 'Positive' if score > 0.5 else 'Negative' , f'Score : {score}'


print(sentiment_analysis('The ')) 




