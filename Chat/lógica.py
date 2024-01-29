from transformers import BertTokenizer, BertForQuestionAnswering
from textwrap import wrap
import torch

contexto = "La ley 21.545 establece la promoción de la inclusión, la atención integral y la protección de los derechos de las personas con trastorno del espectro autista."
pregunta = "¿Qué establece la ley 21.545?"

def cargar_modelo():
    # Cargar el modelo BERT preentrenado y el tokenizador
    modelo = BertForQuestionAnswering.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    tokenizador = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    return modelo, tokenizador

def obtener_respuesta(pregunta, contexto, tokenizador, model):

    # Obtener la respuesta del modelo
    encode = tokenizador.encode_plus(pregunta, contexto, return_tensors='pt')
    input_ids = encode['input_ids']
    outputs = model(**encode)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)+1
    print('\nContexto:')
    print('--------------------------------')
    print('\n'.join(wrap(contexto)))

    respuesta = tokenizador.decode(input_ids[0][answer_start:answer_end])
    return respuesta
    

def Chatbot():
    model, tokenizador = cargar_modelo() 
    
    print("¡Bienvenido al chatbot sobre la ley del espectro autista!")
    print("Puedes hacer preguntas sobre la ley y recibir respuestas.")

    while True:
        pregunta = input("Usuario: ")
        if pregunta.lower() in ["salir", "exit", "fin"]:
            print("¡Hasta luego!")
            break
        else:
            # Obtener la respuesta del modelo
            respuesta = obtener_respuesta(pregunta, contexto, tokenizador, model)
            print("Chatbot:", respuesta)

if __name__ == "__main__":
    Chatbot()

