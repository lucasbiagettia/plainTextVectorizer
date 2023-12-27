import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor

# Conectarse a la base de datos
connection = psycopg2.connect(host="localhost", port="5433", database="database", user="postgres", password="pochovive")

'''
cursor = conn.cursor(cursor_factory=DictCursor)

# Insertar datos en la tabla
metadata_valor = 'valor_metadata'
titulo_valor = 'valor_titulo'
vector_valor = [1.0, 2.0, ..., 400.0]  # Reemplaza con los valores reales del vector

# Construir la consulta SQL de inserción
insert_query = sql.SQL("INSERT INTO nombre_de_tu_tabla (metadata, titulo, vector) VALUES (%s, %s, ST_SetSRID(ST_MakePoint({}), 400))").format(sql.SQL(', ').join(map(sql.Literal, vector_valor)))

# Ejecutar la consulta
cursor.execute(insert_query, (metadata_valor, titulo_valor))

# Confirmar los cambios en la base de datos
conn.commit()

# Cerrar el cursor y la conexión
cursor.close()
conn.close()
'''