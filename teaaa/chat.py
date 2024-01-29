from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


def resumir_texto(texto, max_length=150, model_name="t5-base"):
    
    tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)


    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Tokenizar el texto
    inputs = tokenizer.encode("summarize: " + texto, return_tensors="pt", max_length=512, truncation=True)

    # Generar el resumen
    resumen_ids = model.generate(inputs, max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    resumen = tokenizer.decode(resumen_ids[0], skip_special_tokens=True)

    return resumen

    # ejemplo
texto_a_resumir = """
Según Ayuda Mineduc (2023) la Ley N°21.545, se antecede a partir del:
Año 2016, donde el Congreso declaró de manera transversal el 2 de abril como el Día Nacional de la Concientización sobre el Autismo. En el año 2021 se ingresó el proyecto de Ley TEA a la Cámara de Diputados y Diputadas. Las familias y organizaciones vinculadas a la población TEA fueron parte esencial de la discusión del proyecto, impulsando temáticas y presentando indicaciones ciudadanas, muchas de las cuales fueron incorporadas en el texto. Las organizaciones Fenaut, Fedausch, Vocería autismo quinta, Colectivo Autismo Norte y Fundación EA Femenino Chile, deciden crear la Mesa Interregional por la Ley de Autismo (MILA), que al día de hoy suman más de 200 y que acompañará el proceso de implementación de esta ley.
El 2 de marzo del 2023 se promulgó la ley y entró en vigencia el 10 de marzo del presente año.
"""

resumen = resumir_texto(texto_a_resumir)
print("Texto original:\n", texto_a_resumir)
print("\nResumen:\n", resumen)