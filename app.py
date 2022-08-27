from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('C:/Users/HP/ML/Book_Recommendation_System/model.pkl', "rb"))
Final_data = pickle.load(open('Final_data.pkl','rb'))
Pivot_data = pickle.load(open('Pivot_data.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend')
def recommend_book():
    return render_template('results.html')

@app.route('/recommend_books', methods =['post'])
def recommend():
    book_name = request.form.get('book_name')
    book_id = np.where(Pivot_data.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(Pivot_data.iloc[book_id, :].values.reshape(1, -1), n_neighbors=8)
    #    print(suggestion)

    Recommend_data = []

    for i in range(suggestion.shape[1]):
        Book_List = []
        temp_df = Final_data[Final_data['Book-Title'] == Pivot_data.index[i]]
        Book_List.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        Book_List.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        Book_List.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-L'].values))
        Recommend_data.append(Book_List)
    print(Recommend_data)
    return render_template('results.html', data =Recommend_data)


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app