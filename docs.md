# Indice
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
