# Classificação de e-mails como spam ou não spam

Trabalho N3 de Análise Preditiva (Engenharia de Software, Católica SC). Pipeline MLOps único, executado pelo GitHub Actions, que a cada ciclo incorpora dados novos, refaz a seleção de features por votação, coloca dois modelos para competir e só troca o modelo em produção quando o novo é comprovadamente melhor.

## Objetivo

Classificar e-mails como **spam (1)** ou **não spam (0)** a partir de 57 features numéricas (frequências de palavras, frequências de caracteres e sequências de letras maiúsculas), demonstrando o ciclo de vida completo de um modelo: ingestão incremental, controle de qualidade, seleção de features, competição de modelos, promoção controlada, rastreamento de experimentos, previsões e explicabilidade.

* **Métrica principal:** F1 da classe spam, definida antes de qualquer treino no `config/config.yaml`. Precisão, recall, acurácia e ROC AUC são secundárias.
* **Protocolo:** `StratifiedKFold` com 5 folds, `shuffle=True` e seed 42, igual para todos os modelos.
* **Teste fixo:** 20% dos e-mails, separados uma única vez por grupo de conteúdo (cópias idênticas ficam sempre do mesmo lado) e nunca usados em decisões.

## Fluxo de um ciclo

```
              (uma única vez)
 spambase.data ──► converter_dataset ──► spambase.csv ──► preparar_lotes ──┬─► teste_fixo.csv (20%)
                                                                           ├─► lote_01.csv (60%)
                                                                           └─► lote_02.csv (20%)

 ciclo (GitHub Actions, agendado ou manual)
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

1. **01 ETL:** incorpora o próximo lote sem duplicar registros (`id_email`), diagnostica e trata os dados com regras fixas (sem imputação, para evitar vazamento), mantém os outliers (são sinal de spam, não erro) e roda uma verificação de qualidade que interrompe o pipeline se falhar. Gera o `cycle_id` (data e hora UTC) e o `data_version` (lotes mais hash do treino).
2. **02 Votação:** Random Forest (importância por impureza), LinearSVC e Regressão Logística (|coef| com padronização) ranqueiam as 57 features. Cada método vota no seu top 20; é eleita a feature com 2 votos ou mais, com desempate pela média das posições e mínimo de 10 features. A votação é um transformador (`SeletorPorVotacao`) refeito no treino de cada fold.
3. **03 Competição e promoção:** Modelo A (`RandomForestClassifier`) e Modelo B (`LogisticRegression`) usam o mesmo seletor, o mesmo treino e os mesmos folds. Vence o maior F1 médio. O vencedor só é promovido se tiver F1 >= 0,85 e superar em pelo menos 0,005 o F1 do vigente, reavaliado pelo mesmo protocolo nos dados atuais.
4. **04 Previsões:** carrega o vigente pelo manifesto, prevê no teste fixo, gera a matriz de confusão e as explicações SHAP e só então marca o lote como ingerido.

**Chegada de dados simulada:** o ciclo 1 treina com o `lote_01`, o ciclo 2 adiciona o `lote_02` e, a partir do ciclo 3, não há dados novos, mas a votação e a competição são refeitas. Se qualquer notebook falhar, o job para, nada é commitado e o modelo vigente não muda.

## Estrutura

```
.github/workflows/pipeline_mlops.yml   workflow único (schedule + workflow_dispatch)
config/config.yaml                     todos os parâmetros do projeto
data/raw/original/                     arquivos originais da UCI (.data, .names, .DOCUMENTATION)
data/raw/spambase.csv                  dataset com os nomes oficiais das colunas
data/raw/teste_fixo.csv                teste fixo (20%), nunca usado em decisões
data/raw/lotes/lote_01.csv, lote_02.csv
data/processed/                        treino de cada versão de dados (treino_<data_version>.csv)
notebooks/01_etl.ipynb                 ingestão, diagnóstico, tratamento e qualidade
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
models/modelo_vigente.json             manifesto do modelo em produção e histórico de decisões
models/estado_ingestao.json            lotes já incorporados
models/historico/                      cada modelo treinado (<algoritmo>_<hash>.joblib) e indice_modelos.csv
artifacts/                             ciclo_atual.json e features_eleitas.json
reports/figuras/                       tabelas (CSV) e gráficos (PNG) de cada ciclo, com o cycle_id no nome
reports/figuras/comparacao_ciclos.csv  uma linha por ciclo: lote, features, F1 de A e B, decisão, métricas no teste
evidencias/execucoes/<cycle_id>/       notebooks executados pelo Actions e evento.txt
evidencias/teste_local/                notebooks executados nos testes locais (3 ciclos)
mlflow.db, mlartifacts/                rastreamento do MLflow (versionados)
```

## Execução local

Requer Python 3.12 (mesma linha do Colab e do runner do GitHub Actions). As versões das bibliotecas estão fixadas no `requirements.txt`.

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

Para zerar o estado (lotes ingeridos, vigente, MLflow e figuras) e recomeçar do lote 1:

```bash
python -m src.resetar_estado --confirmar
```

Os scripts de preparação já foram executados e os resultados estão versionados. Só precisam ser rodados de novo se os dados originais mudarem: `python -m src.converter_dataset` e `python -m src.preparar_lotes`.

**No Google Colab:** faça upload de `notebooks/01_etl.ipynb` e use **Executar tudo**. A primeira célula clona o repositório em `/content`, instala o `requirements.txt` e reinicia a sessão se alguma biblioteca mudar de versão (nesse caso, execute de novo). Repita para os notebooks 02, 03 e 04 na mesma sessão e nessa ordem. O que é gerado no Colab fica só na sessão; o estado oficial é o do repositório, atualizado pelo GitHub Actions.

## Execução agendada

O workflow `.github/workflows/pipeline_mlops.yml` tem um único job sequencial:

1. registra o `github.event_name` no log, no resumo da execução e em `evidencias/execucoes/<cycle_id>/evento.txt`;
2. faz checkout, configura o Python 3.12 e instala o `requirements.txt`;
3. executa os notebooks 01 a 04 com `jupyter nbconvert --execute`, salvando as versões executadas em `evidencias/execucoes/<cycle_id>/`;
4. publica com `actions/upload-artifact` os notebooks executados, `mlflow.db`, `mlartifacts/`, `models/`, `reports/figuras/` e `artifacts/`;
5. faz commit do estado atualizado com `[skip ci]` na mensagem (permissão `contents: write`).

* **Gatilhos:** `schedule` todo dia às 09:17 UTC (06:17 em Brasília) e `workflow_dispatch` (botão **Run workflow** na aba Actions).
* **Concorrência:** `concurrency: pipeline-mlops` impede dois ciclos ao mesmo tempo.
* **Para ativar:** em **Settings > Actions > General**, permitir actions e marcar **Read and write permissions** em *Workflow permissions*.

**Execuções registradas:**

| Ciclo | Evento | Decisão |
|---|---|---|
| `20261122T203319Z` | `workflow_dispatch` | promovido (Random Forest, F1 0,9262) |
| `20261123T212753Z` | `schedule` | mantido (desafiante 0,9255 x vigente reavaliado 0,9280) |

A execução agendada pode ser conferida em [actions/runs/45269752956](https://github.com/eduardaramalho/n3-analise-preditiva/actions/runs/45269752956) e em `evidencias/execucoes/20261123T212753Z/evento.txt`. O GitHub pode atrasar disparos agendados: o cron é 09:17 UTC e essa execução começou às 21:27 UTC.

## Consulta ao MLflow

O MLflow usa um banco SQLite (`mlflow.db`) e a pasta `mlartifacts/`, ambos versionados. Como o MLflow grava caminhos absolutos e cada ambiente tem uma raiz diferente (`/content/...` no Colab, `/home/runner/...` no Actions), o `src/rastreamento.py` reescreve esses caminhos para a pasta atual antes de cada uso.

```bash
git pull
python -m src.rastreamento
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Acesse http://127.0.0.1:5000, experimento `n3-spambase-mlops`. Os runs podem ser filtrados pela tag `cycle_id` (ex.: `tags.cycle_id = '20261123T212753Z'`) ou pela tag `etapa`:

| Run | Tag `etapa` | Conteúdo |
|---|---|---|
| `votacao_<cycle_id>` | votacao | regra, rankings, tabela de votos, features eleitas |
| `modelo_a_...`, `modelo_b_...` | competicao | hiperparâmetros, métricas (média, desvio e por fold), features, pipeline serializado |
| `reavaliacao_vigente_<cycle_id>` | reavaliacao_vigente | vigente avaliado nos dados atuais |
| `decisao_<cycle_id>` | decisao | F1 comparados, decisão e motivo |
| `previsoes_<cycle_id>` | previsoes | métricas no teste fixo, matriz de confusão e gráficos SHAP |

Modelos idênticos não são gravados duas vezes: cada pipeline recebe um hash e, se já existir um modelo com o mesmo hash, o run só aponta para ele pela tag `modelo_uri`.

## Reprodução das previsões

```python
import json, joblib, pandas as pd
vigente = json.load(open("models/modelo_vigente.json"))
modelo = joblib.load(vigente["arquivo_modelo"])          # pipeline completo
teste = pd.read_csv("data/raw/teste_fixo.csv")
X = teste.drop(columns=["id_email", "spam"])             # as 57 colunas brutas
prob_spam = modelo.predict_proba(X)[:, 1]
```

O pipeline recebe as 57 colunas e faz sozinho a seleção de features, o escalonamento (nos modelos lineares) e a previsão. O mesmo modelo está no MLflow: `mlflow.sklearn.load_model(vigente["modelo_uri"])`, depois de `configurar_mlflow` de `src/rastreamento.py`. Com a seed 42 e os mesmos dados, o modelo treinado é idêntico.

## Recurso adicional

**Explicabilidade com SHAP (critério 10)**, integrada ao notebook 04 e executada em todo ciclo.

* **Problema:** a Random Forest não é interpretável, e o Spambase tem viés conhecido (palavras como `george` e `hp` indicam e-mail legítimo porque os e-mails vieram de um funcionário da HP).
* **Solução:** `TreeExplainer` para a Random Forest (valores em probabilidade de spam) ou `LinearExplainer` para a Regressão Logística (valores em log-odds), escolhido automaticamente pelo algoritmo do vigente. O fundo do `LinearExplainer` usa amostras do treino, nunca do teste.
* **Saídas:** importância global (barras), beeswarm e waterfall de 3 e-mails do teste (o spam mais confiante, o legítimo mais confiante e o erro mais confiante), salvos em `reports/figuras/` e no run `previsoes_<cycle_id>` do MLflow.
* **Parâmetros:** seção `explicabilidade` do `config.yaml` (300 e-mails explicados no gráfico global, 200 amostras de fundo).

## Dados e licença

**Spambase**, UCI Machine Learning Repository, id 94: 4.601 e-mails, dos quais 1.813 (39,4%) são spam. Os arquivos originais estão em `data/raw/original/`, sem alteração. Licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

HOPKINS, Mark; REEBER, Erik; FORMAN, George; SUERMONDT, Jaap. **Spambase** [dataset]. UCI Machine Learning Repository, 1999. DOI: https://doi.org/10.24432/C53G6X.
