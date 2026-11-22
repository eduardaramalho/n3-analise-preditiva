# Pipeline MLOps: classificação de spam (Trabalho N3, Análise Preditiva, Católica SC)

Pipeline MLOps único, executado diariamente pelo GitHub Actions, que a cada ciclo incorpora dados novos, refaz a seleção de features por votação, coloca dois modelos para competir e só troca o modelo em produção quando o novo é comprovadamente melhor.

## Objetivo

Classificar e-mails como **spam (1)** ou **não spam (0)** a partir de 57 features numéricas de frequência de palavras, caracteres e letras maiúsculas, demonstrando um ciclo de vida completo de modelo: ingestão incremental, controle de qualidade, seleção de features, competição de modelos, promoção controlada, rastreamento de experimentos, previsões e explicabilidade.

**Dataset:** Spambase, UCI Machine Learning Repository, id 94 (HOPKINS et al., 1999), licença CC BY 4.0. São 4.601 e-mails, dos quais 1.813 (39,4%) são spam. Os arquivos originais estão em `data/raw/original/`.

## Arquitetura

```
              (uma única vez)
 spambase.data ──► converter_dataset ──► spambase.csv ──► preparar_lotes ──┬─► teste_fixo.csv (20%)
                                                                           ├─► lote_01.csv (60%)
                                                                           └─► lote_02.csv (20%)

 ciclo (GitHub Actions, diário ou manual)
 ┌──────────────┐   ┌──────────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐
 │ 01 ETL       │──►│ 02 Votação de    │──►│ 03 Competição A x B e    │──►│ 04 Previsões no     │
 │ próximo lote │   │ features (RF,    │   │ promoção controlada      │   │ teste fixo + SHAP   │
 │ diagnóstico  │   │ LinearSVC, LR)   │   │ (vigente reavaliado)     │   │ fecha o ciclo       │
 └──────┬───────┘   └────────┬─────────┘   └────────────┬─────────────┘   └──────────┬──────────┘
        ▼                    ▼                          ▼                            ▼
  treino_<versão>.csv  features_eleitas.json   modelo_vigente.json          comparacao_ciclos.csv
  ciclo_atual.json     run MLflow              historico/ + runs MLflow     estado_ingestao.json
                                                                            run MLflow
 ──► commit do estado atualizado no repositório [skip ci]
```

* **Chegada de dados simulada:** o ciclo 1 treina com o `lote_01`; o ciclo 2 adiciona o `lote_02`; a partir do ciclo 3 não há dados novos, mas a votação e a competição são refeitas.
* **Identificação do ciclo:** cada ciclo tem um `cycle_id` (data e hora UTC, ex.: `20261117T235423Z`) e um `data_version` (lotes incorporados mais hash do treino, ex.: `lotes-01+02_h0ce6906487cc`), compartilhados pelos 4 notebooks em `artifacts/ciclo_atual.json`.
* **Falha para tudo:** se qualquer notebook falhar, o job do Actions para, nada é commitado e o modelo vigente não muda. O lote só é marcado como ingerido no fim do notebook 04.

## Estrutura do repositório

```
.github/workflows/pipeline_mlops.yml   workflow único (schedule + workflow_dispatch)
config/config.yaml                     todos os parâmetros do projeto
data/raw/original/                     arquivos originais da UCI (.data, .names, .DOCUMENTATION)
data/raw/spambase.csv                  dataset com os nomes oficiais das colunas
data/raw/teste_fixo.csv                20% separados uma única vez; nunca usados em decisões
data/raw/lotes/lote_01.csv, lote_02.csv
data/processed/                        treino de cada versão de dados (treino_<data_version>.csv)
notebooks/01_etl.ipynb                 ingestão, diagnóstico, tratamento, qualidade
notebooks/02_votacao_features.ipynb    votação de features
notebooks/03_competicao_promocao.ipynb competição A x B e promoção
notebooks/04_previsoes.ipynb           previsões no teste fixo e SHAP
src/                                   funções reutilizáveis (os notebooks só chamam)
  converter_dataset.py                 .data + .names -> CSV
  preparar_lotes.py                    teste fixo e lotes (executado uma vez)
  ingestao.py                          estado de ingestão, cycle_id, data_version
  preprocessamento.py                  diagnóstico, tratamento e verificação de qualidade
  features.py                          SeletorPorVotacao (transformador do scikit-learn)
  modelagem.py                         pipelines, validação cruzada e métricas
  promocao.py                          eleição, regra de promoção, manifesto e histórico
  explicabilidade.py                   SHAP (TreeExplainer / LinearExplainer)
  rastreamento.py                      MLflow com SQLite portátil entre ambientes
  graficos.py                          padrão visual, tabelas CSV e gráficos PNG
  resetar_estado.py                    zera o estado para testar do zero
models/modelo_vigente.json             manifesto do modelo em produção
models/estado_ingestao.json            lotes já incorporados
models/historico/                      cada modelo treinado (<algoritmo>_<hash>.joblib) e indice_modelos.csv
artifacts/                             ciclo_atual.json e features_eleitas.json
reports/figuras/                       tabelas (CSV) e gráficos (PNG) para o relatório
evidencias/execucoes/<cycle_id>/       notebooks executados pelo Actions e evento.txt
evidencias/teste_local/                notebooks executados nos testes locais (3 ciclos)
mlflow.db, mlartifacts/                rastreamento do MLflow (versionados)
```

## Como executar no Google Colab

Os notebooks rodam sem alteração no Colab, no GitHub Actions e localmente. A primeira célula de cada um detecta o ambiente.

1. Suba o repositório no GitHub (veja "Execução agendada" abaixo).
2. Os notebooks já estão configurados com `REPO_URL = "https://github.com/eduardaramalho/n3-analise-preditiva.git"` na primeira célula de código. Se o repositório mudar de endereço, altere essa variável nos 4 notebooks. Se o repositório for privado, use um token de acesso na URL ou deixe-o público.
3. No Colab: **Arquivo > Fazer upload de notebook** e escolha `notebooks/01_etl.ipynb`.
4. **Ambiente de execução > Executar tudo**. A primeira célula clona o repositório em `/content/<nome-do-repo>`, entra na pasta e instala o `requirements.txt`. Se a instalação trocar a versão de uma biblioteca já carregada, a sessão reinicia sozinha: execute tudo de novo.
5. Repita para os notebooks 02, 03 e 04, **na mesma sessão do Colab** e nessa ordem. Os arquivos gravados por um notebook ficam em `/content/<nome-do-repo>` e são lidos pelo seguinte.
6. Para um novo ciclo, execute de novo do 01 ao 04. Para recomeçar do lote 1, no notebook 01 coloque `RESETAR = True`, execute e depois volte para `False`.

Observação: o que é gerado no Colab fica só na sessão do Colab. O estado "oficial" é o do repositório, atualizado pelo GitHub Actions.

## Execução local

```bash
git clone https://github.com/eduardaramalho/n3-analise-preditiva.git
cd n3-analise-preditiva
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Um ciclo completo:

```bash
for nb in 01_etl 02_votacao_features 03_competicao_promocao 04_previsoes; do
  jupyter nbconvert --to notebook --execute notebooks/$nb.ipynb --output-dir evidencias/execucoes/local || break
done
```

Ou abra os notebooks no Jupyter/VS Code e execute na ordem. Para zerar o estado (lotes ingeridos, vigente, MLflow, figuras):

```bash
python -m src.resetar_estado --confirmar
```

Os scripts de preparação já foram executados e os resultados estão versionados. Só precisam ser rodados de novo se os dados originais mudarem: `python -m src.converter_dataset` e `python -m src.preparar_lotes`.

## Execução agendada (GitHub Actions)

O workflow `.github/workflows/pipeline_mlops.yml` tem um único job sequencial:

1. registra o `github.event_name` no log, no resumo da execução e em `evidencias/execucoes/<cycle_id>/evento.txt` (evidência de que o ciclo foi disparado pelo `schedule`);
2. checkout, Python 3.12 e instalação do `requirements.txt`;
3. executa os notebooks 01, 02, 03 e 04 com `jupyter nbconvert --execute`, salvando as versões executadas em `evidencias/execucoes/<cycle_id>/`;
4. publica com `actions/upload-artifact` os notebooks executados, `mlflow.db`, `mlartifacts/`, `models/`, `reports/figuras/` e `artifacts/`;
5. faz commit do estado atualizado com `[skip ci]` na mensagem (permissão `contents: write`).

* **Gatilhos:** `schedule` todo dia às 09:17 UTC (06:17 em Brasília) e `workflow_dispatch` (botão **Run workflow** na aba Actions).
* **Concorrência:** `concurrency: pipeline-mlops` impede dois ciclos ao mesmo tempo.
* Para ativar: **Settings > Actions > General**, permitir actions e, em *Workflow permissions*, marcar **Read and write permissions**.

## Consulta ao MLflow

O MLflow usa um banco SQLite (`mlflow.db`) e a pasta `mlartifacts/`, ambos versionados. Como o MLflow grava caminhos absolutos e cada ambiente tem uma pasta raiz diferente (`/content/...` no Colab, `/home/runner/...` no Actions), o `src/rastreamento.py` reescreve esses caminhos para a pasta atual antes de cada uso. Para abrir a interface localmente:

```bash
git pull
python -m src.rastreamento
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Acesse http://127.0.0.1:5000, experimento `n3-spambase-mlops`. Cada ciclo gera os runs abaixo, que podem ser filtrados pela tag `cycle_id` (ex.: `tags.cycle_id = '20261117T235423Z'`) ou pela tag `etapa`:

| Run | Tag `etapa` | Conteúdo |
|---|---|---|
| `votacao_<cycle_id>` | votacao | regra, rankings, tabela de votos, features eleitas |
| `modelo_a_...`, `modelo_b_...` | competicao | hiperparâmetros, métricas (média, desvio e por fold), features, pipeline serializado |
| `reavaliacao_vigente_<cycle_id>` | reavaliacao_vigente | vigente avaliado nos dados atuais |
| `decisao_<cycle_id>` | decisao | F1 comparados, decisão e motivo |
| `previsoes_<cycle_id>` | previsoes | métricas no teste fixo, matriz de confusão e gráficos SHAP |

## Reprodução das previsões

```python
import json, joblib, pandas as pd
vigente = json.load(open("models/modelo_vigente.json"))
modelo = joblib.load(vigente["arquivo_modelo"])          # pipeline completo
teste = pd.read_csv("data/raw/teste_fixo.csv")
X = teste.drop(columns=["id_email", "spam"])             # as 57 colunas brutas
prob_spam = modelo.predict_proba(X)[:, 1]
```

O mesmo modelo está no MLflow: `mlflow.sklearn.load_model(vigente["modelo_uri"])` (depois de `configurar_mlflow` de `src/rastreamento.py`). O pipeline recebe as 57 colunas e faz sozinho a seleção de features, o escalonamento e a previsão.

## Saídas para o relatório

Tudo fica em `reports/figuras/`, com o `cycle_id` no início do nome: `<cycle_id>_<notebook>_<conteudo>.csv` ou `.png`. O arquivo `comparacao_ciclos.csv` (e `comparacao_ciclos.png`) acumula uma linha por ciclo: lote novo, features eleitas, F1 de A e B, vencedor, F1 do vigente reavaliado, decisão e motivo, métricas no teste fixo e o evento que disparou o ciclo.

| Notebook | Tabelas (CSV) | Gráficos (PNG) |
|---|---|---|
| 01 | diagnóstico antes e depois, resumo, outliers, tratamentos, verificação de qualidade | outliers x taxa de spam, classes antes e depois |
| 02 | rankings dos 3 métodos, tabela de votos, features eleitas, estabilidade nos folds | top 20 por método, matriz de votos, estabilidade |
| 03 | métricas por fold, comparação A x B, decisão | comparação dos modelos, F1 por fold |
| 04 | métricas no teste, matriz de confusão, relatório por classe, previsões, exemplos, SHAP global e individual | matriz de confusão, SHAP global, beeswarm, waterfall de 3 e-mails |

## Decisões técnicas e justificativas

| Decisão | Justificativa |
|---|---|
| Teste fixo separado uma vez, por grupo de conteúdo | O Spambase tem 391 linhas duplicadas. A divisão mantém cópias idênticas do mesmo lado, para que nenhum e-mail do teste apareça no treino. A estratificação por rótulo e tamanho do grupo mantém a proporção de spam (diferença de até 1 ponto percentual, por causa de um grupo com 69 cópias). |
| Teste fixo nunca usado em decisões | Votação, competição e promoção usam só a validação cruzada no treino. O teste mede o vigente de forma honesta e é comparável entre ciclos. |
| Tratamentos com regras fixas | Faixas válidas vêm da documentação (frequências em [0, 100], maiúsculas >= 1). Ausentes são removidos em vez de imputados, porque imputar fora do Pipeline vazaria estatísticas entre folds. |
| Remoção de duplicatas e rótulos conflitantes no treino | Cópias inflam a validação cruzada (a mesma linha no treino e na validação do fold). E-mails idênticos com rótulos diferentes são contraditórios. |
| Outliers mantidos | Pelo IQR, qualquer ocorrência de palavra vira outlier (mais de 70% dos valores são zero). Os outliers carregam o sinal: entre os outliers de `char_freq_$`, 92% são spam. |
| Votação como transformador dentro do Pipeline | Na validação cruzada, a votação é refeita no treino de cada fold. Escolher as features com todo o treino e depois validar geraria viés de seleção (F1 otimista). No ajuste final, o seletor chega às mesmas features do `features_eleitas.json` (conferido no notebook 03). |
| StandardScaler só nos lineares, dentro do Pipeline | Coeficientes só são comparáveis com as features na mesma escala; árvores não precisam. Ajustado só no treino de cada fold. |
| F1 da classe spam como métrica principal | Equilibra precisão e recall e não é enganado pelo desbalanceamento (60% não spam). |
| Vigente reavaliado nos dados atuais, mesmos folds | Comparar com o F1 antigo do vigente seria comparar dados diferentes. A configuração dele (algoritmo, hiperparâmetros e features) é treinada de novo pelo mesmo protocolo. |
| Limiar de melhoria de 0,005 e F1 mínimo de 0,85 | Diferenças menores que 0,005 estão dentro da variação entre folds e não justificam a troca. O mínimo impede promover um modelo ruim. No ciclo 2 dos testes, o desafiante (0,9255) ficou abaixo do vigente reavaliado (0,9280) e o vigente foi mantido. |
| Seed 42 em tudo | Mesmos folds para todos os modelos e resultados reproduzíveis: com os mesmos dados, o modelo treinado é idêntico. |
| Modelos sem cópias repetidas | Uma Random Forest serializada tem cerca de 12 MB. Cada modelo recebe um hash. Se já existe um modelo idêntico, o run do MLflow só aponta para ele (tag `modelo_uri`) e o histórico não grava de novo. Assim o repositório não cresce a cada ciclo sem dados novos. |
| MLflow em SQLite versionado | Funciona sem servidor no Colab, no Actions e localmente. O histórico fica no próprio repositório. |
| Python 3.12 e versões fixadas | Mesma linha do Colab e do runner. `scipy`, `SQLAlchemy` e `alembic` também foram fixados porque versões mais novas quebravam o `scikit-learn` 1.6 e o MLflow 2.22. |

## Referência

HOPKINS, Mark; REEBER, Erik; FORMAN, George; SUERMONDT, Jaap. **Spambase** [dataset]. UCI Machine Learning Repository, 1999. DOI: https://doi.org/10.24432/C53G6X. Licença CC BY 4.0.
