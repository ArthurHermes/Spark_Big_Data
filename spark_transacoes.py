# instalar as dependências


# Remover a hastag da linha abaixo
# !apt-get install openjdk-11-jdk -y
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PATH"] += os.pathsep + os.path.join(os.environ["JAVA_HOME"], "bin")




#-----------------------#
#        Codigo         #
#-----------------------#


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
import os
import shutil

# ------------------------------------------------
# Caminhos de entrada e saída
# ------------------------------------------------
input_csv = "/content/drive/MyDrive/operacoes_comerciais_inteira.csv"
output_dir = "/content/drive/MyDrive/resultado"

# ------------------------------------------------
# Função auxiliar para remover diretórios de saída antigos
# ------------------------------------------------
def remover_pasta(path):
    if os.path.exists(path):
        shutil.rmtree(path)

# ------------------------------------------------
# Inicialização da sessão Spark
# ------------------------------------------------
spark = SparkSession.builder.appName("TransacoesComerciais").getOrCreate()
sc = spark.sparkContext

# ------------------------------------------------
# 1. LEITURA DO CSV COMO TEXTO (para trabalhar com RDD)
# ------------------------------------------------
text_rdd = sc.textFile(input_csv)
text_rdd = text_rdd.map(lambda line: line.lstrip("\ufeff"))
header = text_rdd.first()
data_rdd = text_rdd.filter(lambda row: row != header)

# ------------------------------------------------
# Função para converter uma linha em lista válida
# ------------------------------------------------
def parse_row(line):
    parts = [p.strip() for p in line.split(";")]
    if len(parts) != 10 or any(p == "" for p in parts):
        return None
    try:
        parts[1] = int(parts[1])  # year
        parts[5] = float(parts[5].replace(",", "."))  # trade_usd
        parts[6] = float(parts[6].replace(",", "."))  # weight_kg
    except:
        return None
    parts[0] = parts[0].strip()  # country_or_area
    parts[4] = parts[4].strip()  # flow
    parts[9] = parts[9].strip()  # category
    return parts

parsed_rdd = data_rdd.map(parse_row).filter(lambda x: x is not None)

# ------------------------------------------------
# Função auxiliar para salvar resultados simples
# ------------------------------------------------
def salvar_resultado(path, conteudo):
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(conteudo) + "\n")

# ================================================================
# 2. CONSULTAS USANDO RDD / PairRDD / SparkSQL
# ================================================================

# ------------------------------------------------
# Q1) Quantas transações envolvem o Brasil (RDD)
# ------------------------------------------------
transacoes_brasil = parsed_rdd.filter(lambda r: r[0].lower() in ("brazil", "brasil"))
q1 = transacoes_brasil.count()
salvar_resultado(f"{output_dir}/q1_transacoes_brasil.txt", q1)

# ------------------------------------------------
# Q2) Mesmo cálculo com PairRDD
# ------------------------------------------------
pair_brasil = parsed_rdd.map(
    lambda r: (r[0].title(), 1) if r[0].lower() in ("brazil", "brasil") else ("Outros", 0)
)
q2 = pair_brasil.filter(lambda kv: kv[0] == "Brazil").reduceByKey(lambda a, b: a + b).collectAsMap()
salvar_resultado(f"{output_dir}/q2_transacoes_brasil_pairrdd.txt", q2)

# ------------------------------------------------
# Q3) Transações do Brasil em 2016 (RDD)
# ------------------------------------------------
q3 = transacoes_brasil.filter(lambda r: r[1] == 2016).count()
salvar_resultado(f"{output_dir}/q3_transacoes_brasil_2016.txt", q3)

# ------------------------------------------------
# Q4) Mesmo cálculo com PairRDD
# ------------------------------------------------
pair_brasil_2016 = parsed_rdd.map(
    lambda r: ((r[0].title(), r[1]), 1)
    if r[0].lower() in ("brazil", "brasil") and r[1] == 2016
    else (("Outros", 0), 0)
)
q4 = pair_brasil_2016.filter(lambda kv: kv[0][0] == "Brazil").reduceByKey(lambda a, b: a + b).collectAsMap()
salvar_resultado(f"{output_dir}/q4_transacoes_brasil_2016_pairrdd.txt", q4)

# ------------------------------------------------
# Q5) Número de transações por fluxo (Flow) e ano (≥2010)
# ------------------------------------------------
q5 = (
    parsed_rdd.map(lambda r: ((r[1], r[4].upper()), 1))
    .filter(lambda kv: kv[0][0] >= 2010)
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda kv: (kv[0][0], kv[0][1]))
)
output_path_q5 = f"{output_dir}/q5_transacoes_por_ano_flow"
remover_pasta(output_path_q5)
q5.map(lambda kv: f"{kv[0]}\t{kv[1]}").coalesce(1).saveAsTextFile(output_path_q5)

# ------------------------------------------------
# Q6) Média de preço (Price) no ano de 2016 (RDD)
# ------------------------------------------------
precos_2016 = parsed_rdd.filter(lambda r: r[1] == 2016).map(lambda r: r[5])
if precos_2016.count() > 0:
    media_2016 = precos_2016.sum() / precos_2016.count()
else:
    media_2016 = 0.0
salvar_resultado(f"{output_dir}/q6_media_preco_2016.txt", media_2016)

# ------------------------------------------------
# Q7) Média de preço (Price) no ano de 2016 (PairRDD)
# ------------------------------------------------
pair_preco_2016 = parsed_rdd.filter(lambda r: r[1] == 2016).map(lambda r: (r[1], (r[5], 1)))
q7 = (
    pair_preco_2016.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    .mapValues(lambda s: s[0] / s[1])
    .collectAsMap()
)
salvar_resultado(f"{output_dir}/q7_media_preco_2016_pairrdd.txt", q7)

# ------------------------------------------------
# Q8) Preço máximo e mínimo por categoria e ano (PairRDD)
# ------------------------------------------------
q8 = (
    parsed_rdd.map(lambda r: ((r[1], r[9].upper()), r[5]))
    .mapValues(lambda p: (p, p))
    .reduceByKey(lambda a, b: (max(a[0], b[0]), min(a[1], b[1])))
    .sortBy(lambda kv: (kv[0][0], kv[0][1]))
)
output_path_q8 = f"{output_dir}/q8_max_min_categoria_ano"
remover_pasta(output_path_q8)
q8.map(lambda kv: f"{kv[0]}\t{kv[1]}").coalesce(1).saveAsTextFile(output_path_q8)

# ------------------------------------------------
# Q9) País com maior valor total de exportação (PairRDD)
# ------------------------------------------------
exportacoes = parsed_rdd.filter(lambda r: r[4].lower() == "export")
soma_export_por_pais = exportacoes.map(lambda r: (r[0].title(), r[5])).reduceByKey(lambda a, b: a + b)
pais_top_export = soma_export_por_pais.takeOrdered(1, key=lambda kv: -kv[1])

if pais_top_export:
    pais, valor = pais_top_export[0]
    q9 = {(pais, "Export"): valor}
else:
    q9 = {}
salvar_resultado(f"{output_dir}/q9_maior_exportacao.txt", q9)

# ------------------------------------------------
# Q10) Preço máximo por país e ano (SparkSQL)
# ------------------------------------------------
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

# ------------------------------------------------
# Q11) Preço mínimo por país e ano (SparkSQL)
# ------------------------------------------------
output_path_q11 = f"{output_dir}/q11_preco_min_pais_ano_ordenado"
remover_pasta(output_path_q11)
q11 = df_clean.groupBy("country_or_area", "year").agg(F.min("trade_usd").alias("min_price"))
q11.orderBy(F.col("year").asc(), F.col("country_or_area").asc()).coalesce(1).write.option(
    "header", True
).option("sep", "\t").mode("overwrite").csv(output_path_q11)

# ------------------------------------------------
# Q12) Transação de exportação com maior preço por kg
# ------------------------------------------------
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
salvar_resultado(f"{output_dir}/q12_maior_preco_por_kg_export.txt", texto_q12)

# ------------------------------------------------
# Finalização
# ------------------------------------------------
spark.stop()
print(f"✅ Execução concluída! Resultados salvos em: {output_dir}")
