from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.optim import AdamW
import torch

# Suponiendo que 'training_data' es tu conjunto de datos que incluye contexto, preguntas y respuestas
# Cada elemento en 'training_data' es un diccionario con 'context', 'question', 'answer'

tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
model = BertForQuestionAnswering.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')


# Datos de entrenamiento (esto es solo un ejemplo, asegúrate de tener tu propio conjunto de datos)
training_data = [
    {'context': 'La ley 21.545 establece la promoción de la inclusión, la atención integral y la protección de los derechos de las personas con trastorno del espectro autista.', 
    'question': '¿Qué establece la ley 21.545?', 'start_position': 1, 'end_position': 20},
    # Agrega más ejemplos según sea necesario
]

inputs = tokenizer(
     [x['context'] for x in training_data], 
     [x['question'] for x in training_data], 
     padding=True, 
     truncation=True, 
     return_tensors='pt', 
     max_length=512
    )

# Suponiendo que tienes las respuestas start y end positions en tus datos
start_positions = torch.tensor([x['start_position'] for x in training_data])
end_positions = torch.tensor([x['end_position'] for x in training_data])

# Crear DataLoader
train_data = TensorDataset(inputs['input_ids'], inputs['attention_mask'], start_positions, end_positions)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

# Configurar el optimizador
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 100

# Ciclo de entrenamiento
model.train()
for epoch in range(num_epochs):  # num_epochs es el número de épocas
    for step, batch in enumerate(train_dataloader):
            input_ids, attention_mask, start_positions, end_positions = batch
            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

# Guardar el modelo entrenado
model.save_pretrained('MiModeloEntrenado')