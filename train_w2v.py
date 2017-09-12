num_features = 400  # Word vector dimensionality
min_word_count = 10  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 20  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import Word2Vec

print("Training model...")
model = Word2Vec(df_all["Text"].apply(lambda x: x.split(' ')), workers=num_workers, \
                 size=num_features, min_count=min_word_count, \
                 window=context, sample=downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)
