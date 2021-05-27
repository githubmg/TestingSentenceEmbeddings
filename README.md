# Sentence embeddings

El paper <a href='https://arxiv.org/abs/1908.10084'>Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks</a> presenta una modificación a la arquitectura de BERT para poder crear embeddings que representen semánticamente a cada oración, una técnica similar a la que se usa para crear embeddings que representen caras. 

A diferencia de los word embeddings que se representan usando entre 50 y 300 dimensiones, los sentence embeddings que entrega la librería <a href='https://pypi.org/project/sentence-transformers/0.3.0/'>sentence transformers</a> son de dimensión 768.


```python
#!pip install -r requirements.txt
```


```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.core.display import HTML
```


```python
model = SentenceTransformer('bert-base-nli-mean-tokens')
```

    Some weights of the model checkpoint at /home/ec2-user/.cache/torch/sentence_transformers/sbert.net_models_bert-base-nli-mean-tokens/0_BERT were not used when initializing BertModel: ['classifier.weight', 'classifier.bias']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


Los word embeddings resultan en representaciones precisas de las palabras, pero no pueden dar cuenta del contexto de las mismas. Como "apple" la fruta y "apple" la marca están representadas por valores idénticos, dependemos de modelos basados en secuencias para poder contextualizarlas y realizar tareas de NLP. Gracias al mecanismo de "atención" que implementa BERT los embeddings de oraciones creados con esta técnica no tienen ese problema. Por eso es interesante ponerlos a prueba con algunos casos específicos. 

La forma de medir la distancia entre este tipo de representaciones es la similitud coseno. Por eso vamos a medir la distancia entre una oración y otras dos, tratando de evaluar si efectivamente el modelo coloca más cerca a la que está <b>SEMÁNTICAMENTE</b> más cerca y no a la que más se parece textualmente. 

## Probando homónimos
La palabra rock de la primera oración está usada en un sentido distinto que en la segunda oración, y por lo tanto la primera debería parecerse más a la tercera.


```python
sentences = [
    "The rolling Stones are rock idols.",
    "Don't throw me a rock.",
    "Iggy Pop is my favourite artist."
]

```


```python
sentence_embeddings = model.encode(sentences)
```


```python
simils = cosine_similarity(
    [sentence_embeddings[0],
    sentence_embeddings[1]]
)
HTML(f'La similitud entre las frases <b>"{sentences[0]}"</b> <br/>  y <b>"{sentences[1]}"</b> , <br/> es de  <b>{simils[0][1]}</b>')
```




La similitud entre las frases <b>"The rolling Stones are rock idols."</b> <br/>  y <b>"Don't throw me a rock."</b> , <br/> es de  <b>0.307941198348999</b>




```python
simils = cosine_similarity(
    [sentence_embeddings[0],
    sentence_embeddings[2]]
)
HTML(f'La similitud entre las frases <b>"{sentences[0]}"</b> <br/>  y <b>"{sentences[2]}"</b> , <br/> es de  <b>{simils[0][1]}</b>')
```




La similitud entre las frases <b>"The rolling Stones are rock idols."</b> <br/>  y <b>"Iggy Pop is my favourite artist."</b> , <br/> es de  <b>0.5321765542030334</b>



En este caso el modelo parece capturar correctamente la diferencia en el significado de rock en el primer y el segundo caso. Si bien la frase sobre Iggy Pop no menciona la palabra rock, habla de música y la similaridad es alta.

## Valor semántico

¿Qué pasa si describimos un sentimiento equivalente sobre un target equivalente pero expresado con otras palabras?


```python
sentences = [
    "I love the capital of Spain.",
    "I like Madrid.",
    "I love the capital of Portugal.",
    "I love the capital of Japan.",
    "I hate the capital of Spain.",
    "I hate Japan"
]
```


```python
sentence_embeddings = model.encode(sentences)
```


```python
outp = ''
for i in range(1,6):
    simils = cosine_similarity(
        [sentence_embeddings[0],
        sentence_embeddings[i]]
    )
    outp += f'La similitud entre las frases <b>"{sentences[0]}"</b> <br/>  y <b>"{sentences[i]}"</b> , <br/> es de  <b>{simils[0][1]}</b><br/><br/>'
HTML(outp)
```




La similitud entre las frases <b>"I love the capital of Spain."</b> <br/>  y <b>"I like Madrid."</b> , <br/> es de  <b>0.8441337943077087</b><br/><br/>La similitud entre las frases <b>"I love the capital of Spain."</b> <br/>  y <b>"I love the capital of Portugal."</b> , <br/> es de  <b>0.8108397126197815</b><br/><br/>La similitud entre las frases <b>"I love the capital of Spain."</b> <br/>  y <b>"I love the capital of Japan."</b> , <br/> es de  <b>0.7268274426460266</b><br/><br/>La similitud entre las frases <b>"I love the capital of Spain."</b> <br/>  y <b>"I hate the capital of Spain."</b> , <br/> es de  <b>0.35674047470092773</b><br/><br/>La similitud entre las frases <b>"I love the capital of Spain."</b> <br/>  y <b>"I hate Japan"</b> , <br/> es de  <b>0.15219761431217194</b><br/><br/>



Estos resultados muestran lo que esperaríamos: aunque la palabra "love" se reemplaza por "like" y "the capital of Spain" se reemplaza por "Madrid" la frase que realmente expresa lo mismo es "I like Madrid". Las expresiones positivas quedan claramente polarizadas respecto de las negativas. También se refleja el hecho de que Japón es un país semánticamente más alejado de España que Portugal. Esto es algo que se podría ver claramente con word embeddings, pero vemos que esta capacidad tampoco se pierde. 

# Parafraseo

¿Qué pasa si expresamos una misma idea modificando fuertemente el orden de las palabras? 



```python
sentences = [
    "Don´t shout at me, John.",
    "Don´t shout at John.",
    "John, stop shouting at me."
]
```


```python
sentence_embeddings = model.encode(sentences)
```


```python
outp = ''
for i in range(1,3):
    simils = cosine_similarity(
        [sentence_embeddings[0],
        sentence_embeddings[i]]
    )
    outp += f'La similitud entre las frases <b>"{sentences[0]}"</b> <br/>  y <b>"{sentences[i]}"</b> , <br/> es de  <b>{simils[0][1]}</b><br/><br/>'
HTML(outp)
```




La similitud entre las frases <b>"Don´t shout at me, John."</b> <br/>  y <b>"Don´t shout at John."</b> , <br/> es de  <b>0.8612778186798096</b><br/><br/>La similitud entre las frases <b>"Don´t shout at me, John."</b> <br/>  y <b>"John, stop shouting at me."</b> , <br/> es de  <b>0.8815945982933044</b><br/><br/>



La primera y la segunda oración son idénticas salvo por una palabra, pero la tercera es la que efectivamente representa el mismo significado y aunque sea con menor confianza los embeddings captan esto correctamente.

### Usos de los sentence embeddings

Estos embeddings son útiles especialmente para escenarios no supervisados: búsquedas de similitud semántica, clustering de oraciones, etc. 



```python

```
