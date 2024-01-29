from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import torch

# Suponiendo que 'training_data' es tu conjunto de datos que incluye contexto, preguntas y respuestas
# Cada elemento en 'training_data' es un diccionario con 'context', 'question', 'answer'

# Configuración del modelo y el tokenizador T5 small
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)  # Utiliza legacy=False para evitar el comportamiento anterior
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Datos de entrenamiento
training_data = [
    {'context': 'La ley 21.545 establece la promoción de la inclusión, la atención integral y la protección de los derechos de las personas con trastorno del espectro autista.', 
    'question': '¿Qué establece la ley 21.545?', 'answer': 'promoción de la inclusión, la atención integral y la protección de los derechos'},
    # Agrega más ejemplos según sea necesario
]

inputs = tokenizer(
    [f"context: {x['context']} question: {x['question']}" for x in training_data],
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=512
)

labels = tokenizer(
    [f"{x['answer']}" for x in training_data],
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=512
)

# Crear DataLoader
train_data = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels['input_ids'], labels['attention_mask'])
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

# Configuración del optimizador
optimizer = AdamW(model.parameters(), lr=2e-5)

# Ciclo de entrenamiento
model.train()
num_epochs = 100
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        input_ids, attention_mask, labels_input_ids, labels_attention_mask = batch
        model.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_input_ids,
            decoder_attention_mask=labels_attention_mask
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Guardar el modelo entrenado
model.save_pretrained('MiModeloEntrenadoT5')
