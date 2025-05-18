
Pipeline completo de classificação de miopatias estruturais, metabólicas e distrofia miotônica. Coloque seu CSV em data e execute conforme instruções.

## Servidor e interface

Para testar rapidamente a API de predição, inicie o servidor FastAPI:

```bash
uvicorn src.api:app --reload
```

Em seguida abra `web/index.html` em seu navegador. Informe o modelo desejado, a lista de features numéricas separadas por vírgula e clique em **Enviar** para obter a predição retornada pela API.
