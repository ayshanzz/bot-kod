import discord
from discord.ext import commands
import os
import torch
from torchvision import models, transforms
from PIL import Image
import requests  


intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

IMAGE_FOLDER = "imagenes_recibidas"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)


model = models.resnet50(pretrained=True)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(LABELS_URL)
if response.status_code == 200:
    imagenet_labels = response.json()
else:
    raise Exception("No se pudieron descargar las etiquetas de ImageNet")

@bot.event
async def on_ready():
    print(f"Bot conectado como {bot.user}")

@bot.command(name="subir_imagen")
async def subir_imagen(ctx):
    if not ctx.message.attachments:
        await ctx.send("No has enviado ninguna imagen. Por favor, adjunta una imagen y vuelve a intentarlo.")
        return

    for attachment in ctx.message.attachments:
        if any(attachment.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif"]):
            save_path = os.path.join(IMAGE_FOLDER, attachment.filename)
            await attachment.save(save_path)  
            await ctx.send("Guardado")

            
            try:
                
                image = Image.open(save_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0)

                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted_idx = outputs.max(1)
                    predicted_label = imagenet_labels[predicted_idx.item()]

                await ctx.send(f"La imagen fue clasificada como: **{predicted_label}**")
            except Exception as e:
                await ctx.send("Ocurrió un error al procesar la imagen. Por favor, intenta de nuevo.")
                print(f"Error al procesar la imagen: {e}")
        else:
            await ctx.send(f"El archivo {attachment.filename} no es una imagen válida.")

bot.run("TOKEN")
