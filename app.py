from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='templates')

# Load the news dataset
df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
news_df = df[df['date'] >= pd.Timestamp(2019, 1, 1)]
X = np.array(news_df.short_description)
text_data = X

# Load the sentence transformer model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(text_data, show_progress_bar=True)
x = np.float16(embeddings)
cos_sim_data = pd.DataFrame(cosine_similarity(x))


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/api/recommend', methods=['POST'])

def recommend():
    data = request.get_json()
    index = data.get('index', None)
    word = data.get('word', None)
    category = data.get('category', None)

    if index is not None:
        index_recomm = cos_sim_data.loc[int(index)].sort_values(ascending=False).index.tolist()[1:6]
        news_recom = news_df['headline'].loc[index_recomm].values.tolist()
    elif word is not None:
        # word_emb = model.encode(word)
        # cos_sim_word = cosine_similarity([word_emb], x)[0]
        recommendations = give_recommendations_word(word,5,category)
        news_recom = recommendations['news'].tolist()  # Convert to list
        index_recomm = recommendations['Index'].tolist()  # Convert to list
    else:
        return jsonify({'error': 'Either index or word must be specified'}), 400

    result = {'news': news_recom, 'index': index_recomm}
    return jsonify(result)

# def give_recommendations_word(cos_sim_word, category=None, print_recommendation=False, print_recommendation_plots=False, print_genres=False):
#     if category is not None:
#         idx = news_df[news_df['category'] == category].index
#         cos_sim_word = cos_sim_word[idx]
#     else:
#         idx = range(len(news_df))

#     index_recomm = np.argsort(cos_sim_word)[::-1][:5]
#     news_recom = news_df['headline'].loc[idx[index_recomm]].values
#     result = {'News': news_recom, 'Index': idx[index_recomm]}
    
#     # ... (previous print_recommendation, print_recommendation_plots, and print_genres code)

#     return result


def give_recommendations_word(query,num_recommendations,print_recommendation=False, print_recommendation_plots=False, print_genres=False,category=None):
    word = query
    word_emb = model.encode(word)


    cos_sim_word = cosine_similarity([word_emb], x)[0]
    if category is not None:
        idx = news_df[news_df['category'] == category].index
        cos_sim_word = cos_sim_word[idx]
    else:
        idx = news_df.index
    index_recomm = np.argsort(cos_sim_word)[::-1][:num_recommendations]
    news_recom = news_df['headline'].loc[index_recomm].values
    category_recom = news_df['category'].iloc[idx[index_recomm]].values
    result = {'news':news_recom, 'Index':index_recomm}
    if print_recommendation:
        # print(f'The input word is: {word}')
        k = 1
        for news, category in zip(news_recom, category_recom):
            # print('The number %i recommended news is this one: %s \n' % (k, category, news))
            k += 1
    if print_recommendation_plots:
        
        k = 1
        for q in range(len(news_recom)):
            plot_q = news_df['short_description'].loc[index_recomm[q]]
            print('The description of the number %i recommended news in the category %s is this one:\n %s \n' % (k, category_recom[q], plot_q))
            k += 1
    if print_genres:
        
        k = 1
        for q in range(len(news_recom)):
            plot_q = news_df['category'].loc[index_recomm[q]]
            print('The category of the number %i recommended news is this one:\n %s \n' % (k, plot_q))
            k += 1
    return result


def collect_feedback_and_improve(query, recommendations, not_relevant_indices):
    # Find the recommendations marked as not relevant
    print("recommendations",recommendations)
    not_relevant = [recommendations['recommendations'][i] for i in not_relevant_indices]
#     print(not_relevant)
    
    if not_relevant:
        # If there are not relevant recommendations, adjust the recommendation process
        # In this case, we increase the number of recommendations to generate
        num_recommendations = len(recommendations['recommendations']) + len(not_relevant)
        print("num_recommendations", num_recommendations)
       
        # Generate new recommendations with the increased number
        new_recommendations = give_recommendations_word(query,num_recommendations,True,True,True)
        print("new_recommendation", new_recommendations)
        # Filter out the recommendations marked as not relevant
        improved_recommendations = [rec for rec in new_recommendations['news'] if rec not in not_relevant]
        recommend_dict={'news':improved_recommendations,'Index':new_recommendations['Index'].tolist()}
        return recommend_dict
   
    else:
        # If there are no not relevant recommendations, the model is already performing well
        return recommendations
@app.route('/not_relevant', methods=['POST'])
def not_relevant():
    data = request.get_json()
    
    query = data['query']
    recommendations = data
    print("data",data)
    not_relevant_indices = data['not_relevant_indices']
    print(not_relevant_indices)
    improved_recommendations = collect_feedback_and_improve(query=query, recommendations=recommendations, not_relevant_indices=not_relevant_indices)
    print(improved_recommendations)

    return jsonify(improved_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
