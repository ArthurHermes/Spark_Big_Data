# ==================== #
#  CONFIGURAÇÃO INICIAL
# ==================== #

# Caso esteja executando no Google Colab, descomente a linha abaixo:
# !apt-get install openjdk-11-jdk -y

import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

# Configuração de ambiente Java para o Spark
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PATH"] += os.pathsep + os.path.join(os.environ["JAVA_HOME"], "bin")

# ==================== #
#  CAMINHOS DE ARQUIVOS
# ==================== #
input_csv = "/content/drive/MyDrive/operacoes_comerciais_inteira.csv"
output_dir = "/content/drive/MyDrive/resultado"

# ================================================================
# FUNÇÕES AUXILIARES
# ================================================================

def remover_pasta(path: str):
    """Remove diretórios antigos para evitar erro de saída duplicada."""
    if os.path.exists(path):
        shutil.rmtree(path)

def salvar_resultado(filename: str, conteudo):
    """Salva resultados simples em arquivos TXT padronizados."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        f.write(str(conteudo) + "\n")

def parse_row(line: str):
    """Realiza o parsing de cada linha do CSV (limpeza e validação)."""
    parts = [p.strip() for p in line.split(";")]
    # Valida estrutura e dados faltantes
    if len(parts) != 10 or any(p == "" for p in parts):
        return None
    try:
        parts[1] = int(parts[1])                   # Year
        parts[5] = float(parts[5].replace(",", "."))  # Price (USD)
        parts[6] = float(parts[6].replace(",", "."))  # Weight
    except:
        return None
    # Normalização de campos importantes
    parts[0] = parts[0].strip()  # Country
    parts[4] = parts[4].strip()  # Flow
    parts[9] = parts[9].strip()  # Category
    return parts

# ================================================================
# INICIALIZAÇÃO DO SPARK
# ================================================================

spark = SparkSession.builder.appName("TransacoesComerciais").getOrCreate()
sc = spark.sparkContext

# ================================================================
# 1) LEITURA DO ARQUIVO CSV COMO RDD
# ================================================================

text_rdd = sc.textFile(input_csv)
text_rdd = text_rdd.map(lambda line: line.lstrip("\ufeff"))  # Remove BOM
header = text_rdd.first()                                   # Captura cabeçalho
data_rdd = text_rdd.filter(lambda row: row != header)       # Remove cabeçalho
parsed_rdd = data_rdd.map(parse_row).filter(lambda x: x is not None)

# ================================================================
# Q1) Número de transações envolvendo o Brasil (RDD)
# ================================================================
transacoes_brasil = parsed_rdd.filter(lambda r: r[0].lower() in ("brazil", "brasil"))
q1 = transacoes_brasil.count()
salvar_resultado("q01_transacoes_brasil.txt", q1)

# ================================================================
# Q2) Número de transações envolvendo o Brasil (PairRDD)
# ================================================================
pair_brasil = parsed_rdd.map(
    lambda r: (r[0].title(), 1) if r[0].lower() in ("brazil", "brasil") else ("Outros", 0)
)
q2 = pair_brasil.filter(lambda kv: kv[0] == "Brazil").reduceByKey(lambda a, b: a + b).collectAsMap()
salvar_resultado("q02_transacoes_brasil_pairrdd.txt", q2)

# ================================================================
# Q3) Número de transações do Brasil em 2016 (RDD)
# ================================================================
q3 = transacoes_brasil.filter(lambda r: r[1] == 2016).count()
salvar_resultado("q03_transacoes_brasil_2016.txt", q3)

# ================================================================
# Q4) Número de transações do Brasil em 2016 (PairRDD)
# ================================================================
pair_brasil_2016 = parsed_rdd.map(
    lambda r: ((r[0].title(), r[1]), 1)
    if r[0].lower() in ("brazil", "brasil") and r[1] == 2016
    else (("Outros", 0), 0)
)
q4 = pair_brasil_2016.filter(lambda kv: kv[0][0] == "Brazil").reduceByKey(lambda a, b: a + b).collectAsMap()
salvar_resultado("q04_transacoes_brasil_2016_pairrdd.txt", q4)

# ================================================================
# Q5) Nº de transações por Flow e Ano (≥ 2010), Flow em maiúsculas (PairRDD)
# ================================================================
q5 = (
    parsed_rdd.map(lambda r: ((r[1], r[4].upper()), 1))
    .filter(lambda kv: kv[0][0] >= 2010)
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda kv: (kv[0][0], kv[0][1]))
)
output_path_q5 = f"{output_dir}/q05_transacoes_por_ano_flow"
remover_pasta(output_path_q5)
q5.map(lambda kv: f"{kv[0]}\t{kv[1]}").coalesce(1).saveAsTextFile(output_path_q5)

# ================================================================
# Q6) Média de Price no ano de 2016 (RDD)
# ================================================================
precos_2016 = parsed_rdd.filter(lambda r: r[1] == 2016).map(lambda r: r[5])
media_2016 = precos_2016.sum() / precos_2016.count() if precos_2016.count() > 0 else 0.0
salvar_resultado("q06_media_preco_2016.txt", media_2016)

# ================================================================
# Q7) Média de Price no ano de 2016 (PairRDD)
# ================================================================
pair_preco_2016 = parsed_rdd.filter(lambda r: r[1] == 2016).map(lambda r: (r[1], (r[5], 1)))
q7 = (
    pair_preco_2016.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    .mapValues(lambda s: s[0] / s[1])
    .collectAsMap()
)
salvar_resultado("q07_media_preco_2016_pairrdd.txt", q7)

# ================================================================
# Q8) Preço máximo e mínimo por Categoria e Ano (PairRDD)
# ================================================================
q8 = (
    parsed_rdd.map(lambda r: ((r[1], r[9].upper()), r[5]))
    .mapValues(lambda p: (p, p))
    .reduceByKey(lambda a, b: (max(a[0], b[0]), min(a[1], b[1])))
    .sortBy(lambda kv: (kv[0][0], kv[0][1]))
)
output_path_q8 = f"{output_dir}/q08_max_min_categoria_ano"
remover_pasta(output_path_q8)
q8.map(lambda kv: f"{kv[0]}\t{kv[1]}").coalesce(1).saveAsTextFile(output_path_q8)

# ================================================================
# Q9) País com maior valor total de Exportação (PairRDD)
# ================================================================
exportacoes = parsed_rdd.filter(lambda r: r[4].lower() == "export")
soma_export_por_pais = exportacoes.map(lambda r: (r[0].title(), r[5])).reduceByKey(lambda a, b: a + b)
pais_top_export = soma_export_por_pais.takeOrdered(1, key=lambda kv: -kv[1])

if pais_top_export:
    pais, valor = pais_top_export[0]
    q9 = {(pais, "Export"): valor}
else:
    q9 = {}
salvar_resultado("q09_maior_exportacao.txt", q9)

# ================================================================
# Q10) Preço máximo por País e Ano (SparkSQL)
# ================================================================
df = spark.read.option("header", True).option("sep", ";").csv(input_csv)
df = df.withColumn("country_or_area", F.regexp_replace("country_or_area", "\\ufeff", ""))

df_clean = (
    df.dropna(subset=["country_or_area", "year", "flow", "trade_usd", "category"])
    .withColumn("year", F.col("year").cast(IntegerType()))
    .withColumn("trade_usd", F.regexp_replace("trade_usd", ",", "").cast(DoubleType()))
    .filter(F.col("year").isNotNull() & F.col("trade_usd").isNotNull())
)

output_path_q10 = f"{output_dir}/q10_preco_max_pais_ano"
remover_pasta(output_path_q10)
q10 = df_clean.groupBy("country_or_area", "year").agg(F.max("trade_usd").alias("max_price"))
q10.coalesce(1).write.option("header", True).option("sep", "\t").mode("overwrite").csv(output_path_q10)

# ================================================================
# Q11) Preço mínimo por País e Ano (SparkSQL)
# ================================================================
output_path_q11 = f"{output_dir}/q11_preco_min_pais_ano_ordenado"
remover_pasta(output_path_q11)
q11 = df_clean.groupBy("country_or_area", "year").agg(F.min("trade_usd").alias("min_price"))
q11.orderBy(F.col("year").asc(), F.col("country_or_area").asc()).coalesce(1).write.option(
    "header", True
).option("sep", "\t").mode("overwrite").csv(output_path_q11)

# ================================================================
# Q12) Transação com maior preço por KG (Flow = Export)
# ================================================================
exportacoes_validas = parsed_rdd.filter(lambda r: r[4].lower() == "export" and r[6] > 0)
preco_por_kg = exportacoes_validas.map(
    lambda r: ((r[1], r[0].title(), r[9].upper()), (r[5] / r[6], r[5], r[6]))
)
maior_preco_kg = preco_por_kg.takeOrdered(1, key=lambda kv: -kv[1][0])

if maior_preco_kg:
    (ano, pais, categoria), (ppk, preco, peso) = maior_preco_kg[0]
    texto_q12 = f"Ano: {ano}, País: {pais}, Categoria: {categoria}, Price/kg: {ppk:.2f}, Price: {preco}, Weight: {peso}"
else:
    texto_q12 = "Nenhuma transação válida de exportação encontrada."
salvar_resultado("q12_maior_preco_por_kg_export.txt", texto_q12)

# ================================================================
# FINALIZAÇÃO
# ================================================================
spark.stop()
print(f"✅ Execução concluída com sucesso! Resultados salvos em: {output_dir}")
