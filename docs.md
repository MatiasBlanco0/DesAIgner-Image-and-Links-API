# Indice
- [Instalacion](#instalacion)
- [Info](#info)
  - [URL](#url-http127001-o-httplocalhost)
  - [Puerto](#puerto-9000)
- [Root](#root)
  - [Path](#path)
  - [Headers](#headers)
  - [Body](#body)
  - [Respuestas](#respuestas)
    - [200](#200)
    - [422](#422)
    - [500](#500)
# Instalacion
**No se cuanto almacenamiento necesitas, pero tene unos GB de mas**
  1. Configurar un virtual enviroment, use pipenv, y activarlo (yo puse el nombre .venv, si vas a hacer un commit y tiene otro nombre agregalo al .gitignore)
  2. Clonar el repo de [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO#hammer_and_wrench-install) y seguir los pasos de instalacion que estan en el README.md (IMPORTANTE: Descargar pytorch (torch en pip) por separado y despues los requirements completos, porque esta troll pip)
  3. Volver a la carpeta de este repo y descargar FastAPI, dotenv, transformers, numpy, PIL, typing extensions y uvicorn. Correr en la terminal `pip install fastapi python-dotenv transformers numpy Pillow typing-extensions uvicorn`
  4. Crear un .env y poner `MERCADO_LIBRE_KEY=key`, la key te la paso por algun lugar seguro
  5. Para iniciar la API corre `uvicorn app:app --host 127.0.0.1 --port 9000` en la carpeta del repo (ESTO VA A DESCARGAR BLIP LA PRIMERA VEZ)
# Info
## URL: http://127.0.0.1 o http://localhost
## Puerto: 9000
# Root
## Path: /
## Headers:
Content-Type: multipart/form-data
## Body:
```javascript
let formData = new FormData();
// Es importante que pongas 'data', porque sino lo guardas en otro lado
formData.append('data', fileField.files[0]); // Esto va en el body
```
## Respuestas
### 200
Todo bien, te da la data
```python
[
  {
    "box": [int, int, int, int],
    "prompt": str,
    "links": [str, str, str]
  },
  ...
]
```
### 422
Pasaste algo mal, fijate como se tiene que enviar (buscalo en google)
```python
{
  "detail": [
    {
      "loc": [str | int],
      "msg": str,
      "type": str
    },
    ...
  ]
}
```
### 500
Fallo algo, mirar la consola xd. Puede ser un problema con la API de mercado libre, la de traduccion o la cague programando.
