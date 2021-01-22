Kommentare in dieser Form:
`commentData = {
                    'id': comment.id,
                    'body': comment.body,
                    'author': comment.author,
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'postId': postData['id']
                }`


Subreddits so:
`postData = {
                'title': post.title,
                'score': post.score,
                'num_comments': post.num_comments,
                'body': post.selftext,
                'id': post.id,
                'url': post.url,
                'permalink': post.permalink,
                'created_utc': post.created_utc
            }`


- python package "requests" für html text von URL
- string concatenation um URL zu ändern
- Daten parsen
- zum Daten speichern Code aus Main verwenden (an anderen Ort speichern)
- PRAW über ID an comments

FANCY
- Function so gestalten, dass Start- und Enddatum variabel angegeben werden können
- Mit Anfangsdatum starten, letzten Post nehmen, dessen Zeit ID für nächstes Startdatum nehmen