#!/usr/bin/env python
# coding: utf-8

# # Nanodegree Engenheiro de Machine Learning
# ## Aprendizagem Não Supervisionada
# ## Projeto 3: Criando Segmentos de Clientela

# Bem-vindo ao terceiro projeto do Nanodegree Engenheiro de Machine Learning! Neste Notebook, alguns modelos de código já foram fornecidos e será seu trabalho implementar funcionalidades adicionais necessárias para completar seu projeto com êxito. Seções que começam com **'Implementação'** no cabeçalho indicam que os blocos de código seguintes vão precisar de funcionalidades adicionais que você deve fornecer. As instruções serão fornecidas para cada seção e as especificações da implementação são marcados no bloco de código com um `'TODO'`. Leia as instruções atentamente!
# 
# Além de implementar códigos, há perguntas que você deve responder relacionadas ao projeto e a sua implementação. Cada seção na qual você responderá uma questão está precedida de um cabeçalho **'Questão X'**. Leia atentamente cada questão e forneça respostas completas nos boxes seguintes que começam com **'Resposta:'**. O envio do seu projeto será avaliado baseado nas suas respostas para cada uma das questões e na implementação que você forneceu.  
# 
# >**Nota:** Células de código e Markdown podem ser executadas utilizando o atalho do teclado **Shift+Enter**. Além disso, células de Markdown podem ser editadas ao dar duplo clique na célula para entrar no modo de edição.

# ## Começando
# 
# Neste projeto, você irá analisar o conjunto de dados de montantes de despesas anuais de vários clientes (reportados em *unidades monetárias*) de diversas categorias de produtos para estrutura interna. Um objetivo deste projeto é melhor descrever a variação de diferentes tipos de clientes que um distribuidor de atacado interage. Isso dará ao distribuidor discernimento sobre como melhor estruturar seu serviço de entrega de acordo com as necessidades de cada cliente.
# 
# O conjunto de dados deste projeto pode ser encontrado no [Repositório de Machine Learning da UCI](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). Para efeitos de projeto, os atributos `'Channel'` e `'Region'` serão excluídos da análise – que focará então nas seis categorias de produtos registrados para clientes.
# 
# Execute o bloco de código abaixo para carregar o conjunto de dados de clientes da distribuidora, junto com algumas das bibliotecas de Python necessárias exigidos para este projeto. Você saberá que o conjunto de dados carregou com êxito se o tamanho do conjunto de dados for reportado.

# In[22]:


# Importe as bibliotecas necessárias para este projeto
import numpy as np
import pandas as pd
from IPython.display import display # Permite o uso de display() para DataFrames

# Importe o código sumplementar para visualização de visuals.py
import visuals as vs

# Mostre matplotlib no corpo do texto (bem formatado no Notebook)
get_ipython().magic(u'matplotlib inline')

# Carregue o conjunto de dados dos clientes da distribuidora de atacado
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"


# ## Explorando os Dados
# Nesta seção, você vai começar a explorar os dados através de visualizações e códigos para entender como cada atributo é relacionado a outros. Você vai observar descrições estatísticas do conjunto de dados, considerando a relevância de cada atributo, e selecionando alguns exemplos de pontos de dados do conjunto de dados que você vai seguir no decorrer do curso deste projeto.
# 
# Execute o bloco de código abaixo para observar as descrições estatísticas sobre o conjunto de dados. Note que o conjunto é compostos de seis categorias importantes de produtos: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'** e **'Delicatessen'** (Perecíveis, Lacticínios, Secos e Molhados, Congelados, Limpeza/Higiene, Padaria/Frios). Considere o que cada categoria representa em termos os produtos que você poderia comprar.

# In[23]:


# Mostre a descrição do conjunto de dados
display(data.describe())


# ### Implementação: Selecionando Amostras
# Para melhor compreensão da clientela e como seus dados vão se transformar no decorrer da análise, é melhor selecionar algumas amostras de dados de pontos e explorá-los com mais detalhes. No bloco de código abaixo, adicione **três** índices de sua escolha para a lista de `indices` que irá representar os clientes que serão acompanhados. Sugerimos que você tente diferentes conjuntos de amostras até obter clientes que variam significativamente entre si.

# In[24]:


# TODO: Selecione três índices de sua escolha que você gostaria de obter como amostra do conjunto de dados
indices = [100, 200, 380]

# Crie um DataFrame das amostras escolhidas
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)


# ### Questão 1
# Considere que a compra total de cada categoria de produto e a descrição estatística do conjunto de dados abaixo para a sua amostra de clientes.  
#  - Que tipo de estabelecimento (de cliente) cada uma das três amostras que você escolheu representa?
# 
# **Dica:** Exemplos de estabelecimentos incluem lugares como mercados, cafés e varejistas, entre outros. Evite utilizar nomes para esses padrões, como dizer *"McDonalds"* ao descrever uma amostra de cliente de restaurante.

# **Resposta:**
# 
# - A primeira amostra parece fazer compras em um estabelecimento com muitos perecíveis, laticíneos e produtos de higiene, portanto, parece ser o cliente de um mercadinho local.
# 
# - A segunda amostra compra muitos produtos "Grocery" e laticíneos, portanto, parece ser o cliente de um supermercado.
# 
# - A terceira amostra possui grande valor de compra em produtos perecíveis, e isso pode significar que são clientes que fazem compras em feiras ou horti-frutis.

# ### Implementação: Relevância do Atributo
# Um pensamento interessante a se considerar é se um (ou mais) das seis categorias de produto são na verdade relevantes para entender a compra do cliente. Dito isso, é possível determinar se o cliente que comprou certa quantidade de uma categoria de produto vai necessariamente comprar outra quantidade proporcional de outra categoria de produtos? Nós podemos determinar facilmente ao treinar uma aprendizagem não supervisionada de regressão em um conjunto de dados com um atributo removido e então pontuar quão bem o modelo pode prever o atributo removido.
# 
# No bloco de código abaixo, você precisará implementar o seguinte:
#  - Atribuir `new_data` a uma cópia dos dados ao remover o atributo da sua escolha utilizando a função `DataFrame.drop`.
#  - Utilizar `sklearn.cross_validation.train_test_split` para dividir o conjunto de dados em conjuntos de treinamento e teste.
#    - Utilizar o atributo removido como seu rótulo alvo. Estabelecer um `test_size` de `0.25` e estebeleça um `random_state`.
#  - Importar uma árvore de decisão regressora, estabelecer um `random_state` e ajustar o aprendiz nos dados de treinamento.
#  - Reportar a pontuação da previsão do conjunto de teste utilizando a função regressora `score`.

# In[25]:


data.head()
# new_data =  data
# new_data.head()
# new_data =  new_data.drop(['Detergents_Paper'], axis=1)
# new_data.head()
# print(new_data)


# In[26]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# TODO: Fazer uma cópia do DataFrame utilizando a função 'drop' para soltar o atributo dado
new_data =  data
new_data =  new_data.drop(['Detergents_Paper'], axis=1)

X = new_data[['Fresh','Milk','Grocery','Frozen','Delicatessen']]
y = (data['Detergents_Paper'])

# TODO: Dividir os dados em conjuntos de treinamento e teste utilizando o atributo dado como o alvo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# TODO: Criar um árvore de decisão regressora e ajustá-la ao conjunto de treinamento
regressor = DecisionTreeRegressor(random_state = 38)
data_higiene = regressor.fit(X_train,y_train)

# TODO: Reportar a pontuação da previsão utilizando o conjunto de teste
score = regressor.score(X_test,y_test)
print(score)


# ### Questão 2
# - Qual atributo você tentou prever?
# - Qual foi a pontuação da previsão reportada?
# - Esse atributo é necessário para identificar os hábitos de compra dos clientes?
# 
# **Dica:** O coeficiente de determinação, `R^2`, é pontuado entre 0 e 1, sendo 1 o ajuste perfeito. Um `R^2` negativo indica que o modelo falhou em ajustar os dados. Se você obter um score baixo para um atributo em particular, isso nos faz acreditar que aquele ponto de atributo é difícil de ser previsto utilizando outros atributos, sendo assim um atributo importante quando considerarmos a relevância.

# **Resposta:**
# 
# - Qual atributo você tentou prever?
# 
#     Detergents_Paper
#     
#     
# - Qual foi a pontuação da previsão reportada?
# 
#     A pontuação foi de 0.49
#     
#     
# - Esse atributo é necessário para identificar os hábitos de compra dos clientes?
# 
#     Devido a não ser um valor alto, acredito que este atributo é sim necessário para identificar os hábitos de compra dos clientes

# ### Visualizando a Distribuição de Atributos
# Para entender melhor o conjunto de dados, você pode construir uma matriz de dispersão de cada um dos seis atributos dos produtos presentes nos dados. Se você perceber que o atributo que você tentou prever acima é relevante para identificar um cliente específico, então a matriz de dispersão abaixo pode não mostrar nenhuma relação entre o atributo e os outros. Da mesma forma, se você acredita que o atributo não é relevante para identificar um cliente específico, a matriz de dispersão pode mostrar uma relação entre aquele e outros atributos dos dados. Execute o bloco de código abaixo para produzir uma matriz de dispersão.

# In[27]:


# Produza uma matriz de dispersão para cada um dos pares de atributos dos dados
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Questão 3:
# 
# - Usando a matriz de dispersão como referência, discuta a distribuição da base de dados. Elabore sua resposta considerando a normalidade, _outliers_, a grande quantidade de pontos próximo de 0 e outras coisas que julgar importante. Se necessário, você pode realizar outros plots para complementar sua explicação.
# - Há algum par de atributos que mostra algum grau de correlação?
# - Como isso confirma ou nega a suspeita sobre relevância do atributo que você tentou prever?
# - Como os dados desses atributos são distribuidos?
# 
# **Dica:** Os dados são distribuídos normalmente? Onde a maioria dos pontos estão? Você pode usar [corr()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) para ver a correlação dos atributos e visualiza-los utilizando um [heatmap](http://seaborn.pydata.org/generated/seaborn.heatmap.html)(os dados que alimentam o heatmap seriam as correlações, por exemplo `data.corr()`)

# **Resposta:**
# 
# - Usando a matriz de dispersão como referência, discuta a distribuição da base de dados. Elabore sua resposta considerando a normalidade, _outliers_, a grande quantidade de pontos próximo de 0 e outras coisas que julgar importante. Se necessário, você pode realizar outros plots para complementar sua explicação.
# 
#         A base de dados em questão parece ter dados muito semelhantes, todos os atributos parecem ser independentes entre eles e, portanto, necessários para identificar diferentes segmentos de clientes.
# 
# - Há algum par de atributos que mostra algum grau de correlação?
# 
#         Não há um par de atributos que haja uma correlação muito significativa, apenas superficiais.
#         
#         
# - Como isso confirma ou nega a suspeita sobre relevância do atributo que você tentou prever?
# 
#         Isso confirma que não há uma grande relação entre o atributo previsto com os outros atributos
#         
# 
# - Como os dados desses atributos são distribuidos?
# 
#         Todos os dados parecem ter uma distribuição parecida, todos muito próximos de 0, com poucos outliers e todos os outliers parecem ter o mesmo volume em todos os atributos.

# ## Pré-processamento de Dados
# Nesta seção, você irá pré-processar os dados para criar uma melhor representação dos clientes ao executar um escalonamento dos dados e detectando os discrepantes. Pré-processar os dados é geralmente um passo fundamental para assegurar que os resultados obtidos na análise são importantes e significativos.

# ### Implementação: Escalonando Atributos
# Se os dados não são distribuídos normalmente, especialmente se a média e a mediana variam significativamente (indicando um grande desvio), é quase sempre [apropriado] ](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) aplicar um escalonamento não linear – particularmente para dados financeiros. Uma maneira de conseguir escalonar dessa forma é utilizando o [ teste Box-Cox](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), que calcula o melhor poder de transformação dos dados, que reduzem o desvio. Uma abordagem simplificada que pode funcionar na maioria dos casos seria aplicar o algoritmo natural.
# 
# No bloco de código abaixo, você vai precisar implementar o seguinte:
#  - Atribua uma cópia dos dados para o `log_data` depois de aplicar um algoritmo de escalonamento. Utilize a função `np.log` para isso.
#  - Atribua uma cópia da amostra do dados para o `log_samples` depois de aplicar um algoritmo de escalonamento. Novamente, utilize o `np.log`.

# In[28]:


# TODO: Escalone os dados utilizando o algoritmo natural
log_data = np.log(data)

# TODO: Escalone a amostra de dados utilizando o algoritmo natural
log_samples = np.log(log_data)

# Produza uma matriz de dispersão para cada par de atributos novos-transformados
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Observação
# Após aplicar o algoritmo natural para o escalonamento dos dados, a distribuição para cada atributo deve parecer mais normalizado. Para muitos pares de atributos, você vai precisar identificar anteriormente como sendo correlacionados, observe aqui se essa correlação ainda está presente (e se está mais forte ou mais fraca que antes).
# 
# Execute o código abaixo para ver como a amostra de dados mudou depois do algoritmo natural ter sido aplicado a ela.

# In[29]:


# Mostre a amostra dados log-transformada
display(log_samples)


# ### Implementação: Detecção de valores atípicos (_Outlier_)
# Identificar dados discrepantes é extremamente importante no passo de pré-processamento de dados de qualquer análise. A presença de discrepantes podem enviesar resultados que levam em consideração os pontos de dados. Há muitas "regras básicas" que constituem um discrepante em um conjunto de dados. Aqui usaremos [o Método Turco para identificar valores atípicos](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): Um *passo do discrepante* é calculado 1,5 vezes a variação interquartil (IQR). Um ponto de dados com um atributo que está além de um passo de um discrepante do IQR para aquele atributo, ele é considerado anormal.
# 
# No bloco de código abaixo, você vai precisar implementar o seguinte:
#  - Atribuir o valor do 25º percentil do atributo dado para o `Q1`. Utilizar `np.percentile` para isso.
#  - Atribuir o valor do 75º percentil do atributo dado para o `Q3`. Novamente, utilizar `np.percentile`.
#  - Atribuir o cálculo de um passo do discrepante do atributo dado para o `step`.
#  - Remover opcionalmentos os pontos de dados do conjunto de dados ao adicionar índices à lista de `outliers`.
# 
# **NOTA:** Se você escolheu remover qualquer discrepante, tenha certeza que a amostra de dados não contém nenhum desses pontos!  
#  Uma vez que você executou essa implementação, o conjunto de dado será armazenado na variável `good_data`!

# In[30]:


for feature in log_data.keys():
    print(feature)


# In[31]:


# Para cada atributo encontre os pontos de dados com máximos valores altos e baixos
for feature in log_data.keys():
    
    # TODO: Calcule Q1 (25º percentil dos dados) para o atributo dado
    Q1 = np.percentile(log_data[feature],25)
    
    # TODO: Calcule Q3 (75º percentil dos dados) para o atributo dado
    Q3 = np.percentile(log_data[feature],75)
    
    # TODO: Utilize a amplitude interquartil para calcular o passo do discrepante (1,5 vezes a variação interquartil)
    step = 1.5*(Q3-Q1)
    
    # Mostre os discrepantes
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
# OPCIONAL: Selecione os índices dos pontos de dados que você deseja remover
outliers  = [65,66,75,128,154]

# Remova os valores atí, caso nenhum tenha sido especificado
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


# ### Questão 4
# - Há alguns pontos de dado considerados discrepantes de mais de um atributo baseado na definição acima?
# - Esses pontos de dados deveriam ser removidos do conjunto?
# - Se qualquer ponto de dados foi adicionado na lista `outliers` para ser removido, explique por quê.

# **Resposta:**
# 
# - Há alguns pontos de dado considerados discrepantes de mais de um atributo baseado na definição acima?
# 
#     Sim, e mais de um.
#     
#     
# - Esses pontos de dados deveriam ser removidos do conjunto?
# 
#     Deveriam, já que se repetem, eles estão alterando a análise em diversos atributos.
#     
# 
# - Se qualquer ponto de dados foi adicionado na lista `outliers` para ser removido, explique por quê.
# 
#     O ponto de índice 75, 66, 65, 128 e 154 já que se repetem em diversos atributos.

# ## Transformação de Atributo
# Nesta seção, você irá utilizar a análise de componentes principais (PCA) para elaborar conclusões sobre a estrutura subjacente de dados de clientes do atacado. Dado que ao utilizar a PCA em conjunto de dados calcula as dimensões que melhor maximizam a variância, nós iremos encontrar quais combinações de componentes de atributos melhor descrevem os consumidores.

# ### Implementação: PCA
# 
# Agora que os dados foram escalonados em uma distribuição normal e qualquer discrepante necessário foi removido, podemos aplicar a PCA na `good_data` para descobrir qual dimensão dos dados melhor maximizam a variância dos atributos envolvidos. Além de descobrir essas dimensões, a PCA também irá reportar a *razão da variância explicada* de cada dimensão – quanta variância dentro dos dados é explicada pela dimensão sozinha. Note que o componente (dimensão) da PCA pode ser considerado como um novo "feature" do espaço, entretanto, ele é uma composição do atributo original presente nos dados.
# 
# No bloco de código abaixo, você vai precisar implementar o seguinte:
#  - Importar o `sklearn.decomposition.PCA` e atribuir os resultados de ajuste da PCA em seis dimensões com o `good_data` para o `pca`.
#  - Aplicar a transformação da PCA na amostra de log-data `log_samples` utilizando `pca.transform`, e atribuir os resultados para o `pca_samples`.

# In[32]:


from sklearn.decomposition import PCA

# TODO: Aplique a PCA ao ajustar os bons dados com o mesmo número de dimensões como atributos
pca = PCA(n_components=6)
pca.fit((good_data))

# TODO: Transforme a amostra de data-log utilizando o ajuste da PCA acima
pca_samples = pca.transform(log_samples)

# Gere o plot dos resultados da PCA
pca_results = vs.pca_results(good_data, pca)


# ### Questão 5
# - Quanta variância nos dados é explicada **no total** pelo primeiro e segundo componente principal?
# - Quanta variância nos dados é explicada pelos quatro primeiros componentes principais?
# - Utilizando a visualização fornecida acima, discuta quais das quatro primeiras dimensões que melhor representam em termos de despesas dos clientes. Explique qual das quatro representa melhor em termos de consumo dos clientes.
# 
# **Dica:** Uma melhora positiva dentro de uma dimensão específica corresponde a uma *melhora* do atributos de *pesos-positivos* e uma *piora* dos atributos de *pesos-negativos*. A razão de melhora ou piora é baseada nos pesos de atributos individuais.

# **Resposta:**
# - Quanta variância nos dados é explicada **no total** pelo primeiro e segundo componente principal?
# 
# 
#     No primeiro e segundo podemos ver uma variãncia de 0.71, onde quase todos os atributos tem resultado negativo.
#     
# 
# - Quanta variância nos dados é explicada pelos quatro primeiros componentes principais?
# 
# 
#     Nos quatro primeiros, podemos ver que Frozen e Delicatessen são destaques, principalmente na dimensão 4, onde frozen possui um valor bem alto, totalizando uma variância de 0.93.
#     
#     
# - Utilizando a visualização fornecida acima, discuta quais das quatro primeiras dimensões que melhor representam em termos de despesas dos clientes. Explique qual das quatro representa melhor em termos de consumo dos clientes.
# 
#     
#     Para a questão de despesa dos clientes, as Dimensões 4 e 3 são as melhores opções, já que ao somar os gastos é possível notar que terá o menos variação no reulstado. Já para o consumo dos clientes, as melhores opções seriam a 1 e 2 que melhor demonstra a distribuição de consumo entre diferentes atributos.
#   

# ### Observação
# Execute o código abaixo para ver como a amostra de log transformado mudou depois de receber a transformação da PCA aplicada a ele em seis dimensões. Observe o valor numérico para as quatro primeiras dimensões para os pontos da amostra. Considere se isso for consistente com sua interpretação inicial dos pontos da amostra.

# In[33]:


# Exiba a amostra de log-data depois de aplicada a tranformação da PCA
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# ### Implementação: Redução da Dimensionalidade
# Ao utilizar um componente principal de análise, um dos objetivos principais é reduzir a dimensionalidade dos dados – na realidade, reduzindo a complexidade do problema. Redução de dimensionalidade tem um custo: Poucas dimensões utilizadas implicam em menor variância total dos dados que estão sendo explicados. Por causo disso, a *taxa de variância explicada cumulativa* é extremamente importante para saber como várias dimensões são necessárias para o problema. Além disso, se uma quantidade significativa de variância é explicada por apenas duas ou três dimensões, os dados reduzidos podem ser visualizados depois.
# 
# No bloco de código abaixo, você vai precisar implementar o seguinte:
#  - Atribuir os resultados de ajuste da PCA em duas dimensões com o `good_data` para o `pca`.
#  - Atribuir a tranformação da PCA do `good_data` utilizando `pca.transform`, e atribuir os resultados para `reduced_data`.
#  - Aplicar a transformação da PCA da amostra do log-data `log_samples` utilizando `pca.transform`, e atribuindo os resultados ao `pca_samples`.

# In[34]:


# TODO: Aplique o PCA ao ajusta os bons dados com apenas duas dimensões
pca = PCA(n_components=2)
pca.fit(good_data)

# TODO: Transforme os bons dados utilizando o ajuste do PCA acima
reduced_data = pca.transform(good_data)

# TODO: Transforme a amostre de log-data utilizando o ajuste de PCA acima
pca_samples = pca.transform(log_samples)

# Crie o DataFrame para os dados reduzidos
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# ### Observação
# Execute o código abaixo para ver como a amostra de dados do log-transformado mudou depois de receber a transformação do PCA aplicada a ele em apenas duas dimensões. Observe como os valores das duas primeiras dimensões permanessem constantes quando comparados com a transformação do PCA em seis dimensões.

# In[35]:


# Exiba a amostra de log-data depois de aplicada a transformação da PCA em duas dimensões
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# ## Visualizando um Biplot
# Um biplot é um gráfico de dispersão onde cada ponto é representado por sua pontuação junto das componentes principais. Os eixos são as componentes principais (nesse caso, `Dimension 1` e `Dimenson 2`). Além disso, o biplot mostra a projeção dos atributos originais junto das componentes. Um biplot pode nos ajudar a interpretar a redução da dimensionalidade dos dados e descobrir relacionamentos entre as componentes principais e os atributos originais.
# 
# Execute a célula abaixo para produzir um biplot com os dados de dimensionalidade reduzida.

# In[36]:


# Create a biplot
vs.biplot(good_data, reduced_data, pca)


# ## Clustering
# 
# Nesta seção, você irá escolher utilizar entre o algoritmo de clustering K-Means ou o algoritmo de clustering do Modelo de Mistura Gaussiano para identificar as várias segmentações de clientes escondidos nos dados. Então você irá recuperar pontos de dados específicos do cluster para entender seus significados ao transformá-los de volta em suas dimensões e escalas originais. 

# ### Questão 6
# - Quais são as vantagens de utilizar o algoritmo de clustering K-Means?
# - Quais são as vantagens de utilizar o algoritmo de clustering do Modelo de Mistura Gaussiano?
# - Dadas as suas observações até agora sobre os dados de clientes da distribuidora, qual dos dois algoritmos você irá utilizar e por quê.
# 
# **Dica: **Pense na diferença entre os clusters mais próximos ou mais isolados.

# **Resposta:**
# 
# - Quais são as vantagens de utilizar o algoritmo de clustering K-Means?
# 
#     
#     Dentre as vantagens, está o fato de ser rápido e não usar tanto processamento computacional, além de ser ótimo com datasets muito grandes e produzir clusters bem definidos.
#     
#     
# - Quais são as vantagens de utilizar o algoritmo de clustering do Modelo de Mistura Gaussiano?
# 
# 
#     A maior vantagem do Modelo de Mistura Gaussiano é sua "maleabilidade", já que pode gerar clusters em diversos formatos, além do esférico também produzido pelo K-Means.
#     
#     
# - Dadas as suas observações até agora sobre os dados de clientes da distribuidora, qual dos dois algoritmos você irá utilizar e por quê.
# 
# 
#     Irei usar o Modelo de Mistura Gaussiano devido principalmente a análise que fizemos do biplot, onde vemos que os dados seguem uma tendência não muito uniforme, não muito esférica.

# ### Implementação: Criando Clusters
# Dependendo do problema, o número de clusters que você espera que estejam nos dados podem já ser conhecidos. Quando um número de clusters não é conhecido *a priori*, não há garantia que um dado número de clusters melhor segmenta os dados, já que não é claro quais estruturas existem nos dados – se existem. Entretanto, podemos quantificar a "eficiência" de um clustering ao calcular o *coeficiente de silhueta* de cada ponto de dados. O [coeficiente de silhueta](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) para um ponto de dado mede quão similar ele é do seu cluster atribuído, de -1 (não similar) a 1 (similar). Calcular a *média* do coeficiente de silhueta fornece um método de pontuação simples de um dado clustering.
# 
# No bloco de código abaixo, você vai precisar implementar o seguinte:
#  - Ajustar um algoritmo de clustering para o `reduced_data` e atribui-lo ao `clusterer`.
#  - Prever o cluster para cada ponto de dado no `reduced_data` utilizando o `clusterer.predict` e atribuindo eles ao `preds`.
#  - Encontrar os centros do cluster utilizando o atributo respectivo do algoritmo e atribuindo eles ao `centers`.
#  - Prever o cluster para cada amostra de pontos de dado no `pca_samples` e atribuindo eles ao `sample_preds`.
#  - Importar sklearn.metrics.silhouette_score e calcular o coeficiente de silhueta do `reduced_data` contra o do `preds`.
#    - Atribuir o coeficiente de silhueta para o `score` e imprimir o resultado.

# In[37]:


# TODO: Aplique o algoritmo de clustering de sua escolha aos dados reduzidos 
from sklearn import mixture
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy

clusterer = mixture.GaussianMixture(n_components=2, covariance_type='full', random_state = 42).fit(reduced_data)

# TODO: Preveja o cluster para cada ponto de dado
preds = clusterer.predict(reduced_data)

# TODO: Ache os centros do cluster
centers = clusterer.means_

# TODO: Preveja o cluster para cada amostra de pontos de dado transformados
sample_preds = clusterer.predict(pca_samples)

# TODO: Calcule a média do coeficiente de silhueta para o número de clusters escolhidos
score = silhouette_score(reduced_data, preds)

print(numpy.mean(score))


# ### Questão 7
# - Reporte o coeficiente de silhueta para vários números de cluster que você tentou.
# - Dentre eles, qual a quantidade de clusters que tem a melhor pontuação de silhueta?

# **Resposta:**
# 
# - Reporte o coeficiente de silhueta para vários números de cluster que você tentou.
#     
#     
#     Para 2: 0.415
#     Para 3: 0.238
#     para 4: 0.332
#     para 6: 0.333
#     
# 
# - Dentre eles, qual a quantidade de clusters que tem a melhor pontuação de silhueta?
# 
#     
#     O melhor score foi obtido com 2 clusters.

# ### Visualização de Cluster
# Uma vez que você escolheu o número ótimo de clusters para seu algoritmo de clustering utilizando o método de pontuação acima, agora você pode visualizar os resultados ao executar o bloco de código abaixo. Note que, para propósitos de experimentação, é de bom tom que você ajuste o número de clusters para o seu algoritmo de cluster para ver várias visualizações. A visualização final fornecida deve, entretanto, corresponder com o número ótimo de clusters. 

# In[38]:


# Mostre os resultados do clustering da implementação
vs.cluster_results(reduced_data, preds, centers, pca_samples)


# ### Implementação: Recuperação de Dados
# Cada cluster apresentado na visualização acima tem um ponto central. Esses centros (ou médias) não são especificamente pontos de dados não específicos dos dados, em vez disso, são *as médias* de todos os pontos estimados em seus respectivos clusters. Para o problema de criar segmentações de clientes, o ponto central do cluster corresponde *a média dos clientes daquele segmento*. Já que os dados foram atualmente reduzidos em dimensões e escalas por um algoritmo, nós podemos recuperar a despesa representativa do cliente desses pontos de dados ao aplicar transformações inversas.
# 
# No bloco de código abaixo, você vai precisar implementar o seguinte:
#  - Aplicar a transformação inversa para o `centers` utilizando o `pca.inverse_transform`, e atribuir novos centros para o `log_centers`.
#  - Aplicar a função inversa do `np.log` para o `log_centers` utilizando `np.exp`, e atribuir os verdadeiros centros para o `true_centers`.
# 

# In[39]:


# TODO: Transforme inversamento os centros
log_centers = pca.inverse_transform(centers)

# TODO: Exponencie os centros
true_centers = np.exp(log_centers)

# Mostre os verdadeiros centros
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)


# ### Questão 8
# - Considere o gasto total de compra de cada categoria de produto para os pontos de dados representativos acima e reporte a descrição estatística do conjunto de dados no começo do projeto. Qual conjunto de estabelecimentos cada segmentação de clientes representa?
# 
# **Dica:** Um cliente que é atribuído ao `'Cluster X'` deve se identificar melhor com os estabelecimentos representados pelo conjunto de atributos do `'Segment X'`. Pense no que cada segmento representa em termos do ponto de atributo escolhido.

# **Resposta:**
# 
# O segmento 0 parece representar o cliente que compra em um mercadinho local, como dito no início do projeto.
# 
# Já o Segmento 1, parece com um cliente de suoermercado, pela distribuição quase igual, porém tendendo a "Grocery".

# ### Questão 9
# - Para cada amostra de ponto, qual segmento de cliente da **Questão 8** é melhor representado?
# - As previsões para cada amostra de ponto são consistentes com isso?
# 
# Execute o bloco de códigos abaixo para saber a previsão de segmento para cada amostra de ponto.

# In[40]:


# Mostre as previsões
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred


# **Resposta:**
# 
# O segmento de cliente melhor representado é o de segmento 0.
# 
# As previsões são consistentes com isso, por mostrar uma maior diversidade de compra.

# ## Conclusão

# Nesta seção final, você irá investigar maneiras de fazer uso dos dados que estão em clusters. Primeiro você vai considerar quais são os diferentes grupos de clientes, a **segmentação de clientes**, que pode ser afetada diferentemente por um esquema de entrega específico. Depois, você vai considerar como dar um rótulo para cada cliente (qual *segmento* aquele cliente pertence), podendo fornecer atributos adicionais sobre os dados do cliente. Por último, você vai comparar a **segmentação de clientes** com uma variável escondida nos dados, para ver se o cluster identificou certos tipos de relação.

# ### Questão 10
# Empresas sempre irão executar os [testes A/B](https://en.wikipedia.org/wiki/A/B_testing) ao fazer pequenas mudanças em seus produtos ou serviços para determinar se ao fazer aquela mudança, ela afetará seus clientes de maneira positiva ou negativa. O distribuidor de atacado está considerando mudar seu serviço de entrega de atuais 5 dias por semana para 3 dias na semana. Mas o distribuidor apenas fará essa mudança no sistema de entrega para os clientes que reagirem positivamente.
# - Como o distribuidor de atacado pode utilizar a segmentação de clientes para determinar quais clientes, se há algum, que serão alcançados positivamente à mudança no serviço de entrega?
# 
# **Dica:** Podemos supor que as mudanças afetam todos os clientes igualmente? Como podemos determinar quais grupos de clientes são os mais afetados?

# **Resposta:**
# 
# - Como o distribuidor de atacado pode utilizar a segmentação de clientes para determinar quais clientes, se há algum, que serão alcançados positivamente à mudança no serviço de entrega?
# 
#     
#     Existem diversas formas de resolver essa questão, como por exemplo, considerar o tipo de produto a ser entregue mais vezes. Nos segmentos onde existem mais alimentos perecíveis, frescos e laticíneos, provavelmente seria melhor manter o serviço de entrega de atuais 5 dias, já nos outros segmentos pode ser possível entregar apenas 3 dias na semana devido a se tratar de produtos com maior tempo de vida.
#     
#     Uma outra forma de enxergar essa solução, seria identificar em quais segmentos há maior volume de compra, assim, fazendo com que os produtos acabem mais rápido e que necessitem de entregas mais vezes na semana, precisando também manter os 5 dias atuais.

# ### Questão 11
# A estrutura adicional é derivada dos dados não rotulados originalmente quando utilizado as técnicas de clustering. Dado que cada cliente tem um **segmento de cliente** que melhor se identifica (dependendo do algoritmo de clustering aplicado), podemos considerar os *segmentos de cliente* como um **atributo construído (engineered)** para os dados. Assumindo que o distribuidor de atacado adquiriu recentemente dez novos clientes e cada um deles forneceu estimativas dos gastos anuais para cada categoria de produto. Sabendo dessas estimativas, o distribuidor de atacado quer classificar cada novo cliente em uma **segmentação de clientes** para determinar o serviço de entrega mais apropriado.  
# - Como o distribuidor de atacado pode rotular os novos clientes utilizando apenas a estimativa de despesas com produtos e os dados de **segmentação de clientes**.
# 
# **Dica:** Um aprendiz supervisionado pode ser utilizado para treinar os clientes originais. Qual seria a variável alvo?

# **Resposta:**
# 
# Neste caso o distribuidor precisaria rotular os novos clientes principalmente de acordo com os tipos de produtos com maior volume de venda, assim levando em conta o produto, poderá decidir a necessidade desse cliente. Voltando na questão anterior, se for um cliente com alto consumo de perecíveis, necessitará de entrega 5 dias na semana, e a mesma coisa para um cliente que esgota rapidamente seu estoque de produtos.

# ### Visualizando Distribuições Subjacentes
# 
# No começo deste projeto, foi discutido que os atributos `'Channel'` e `'Region'` seriam excluídos do conjunto de dados, então as categorias de produtos do cliente seriam enfatizadas na análise. Ao reintroduzir o atributo `'Channel'` ao conjunto de dados, uma estrutura interessante surge quando consideramos a mesma redução de dimensionalidade da PCA aplicada anteriormente no conjunto de dados original.
# 
# Execute o código abaixo para qual ponto de dados é rotulado como`'HoReCa'` (Hotel/Restaurante/Café) ou o espaço reduzido `'Retail'`. Al´´em disso, você vai encontrar as amostras de pontos circuladas no corpo, que identificará seu rótulo.

# In[41]:


# Mostre os resultados do clustering baseado nos dados do 'Channel'
vs.channel_results(reduced_data, outliers, pca_samples)


# ### Questão 12
# - Quão bom é o algoritmo de clustering e o números de clusters que você escolheu comparado a essa distribuição subjacente de clientes de Hotel/Restaurante/Café a um cliente Varejista?
# - Há segmentos de clientes que podem ser classificados puramente como 'Varejistas' ou 'Hotéis/Restaurantes/Cafés' nessa distribuição?
# - Você consideraria essas classificações como consistentes comparada a sua definição de segmentação de clientes anterior?*

# **Resposta:**
# 
# - Quão bom é o algoritmo de clustering e o números de clusters que você escolheu comparado a essa distribuição subjacente de clientes de Hotel/Restaurante/Café a um cliente Varejista?
# 
#     Ao usar o atributo "Channel", podemos ver que os clusters classificaram diversos pontos erroneamente, portanto, se tornou um algortimo bem pior.
#     
#     
# - Há segmentos de clientes que podem ser classificados puramente como 'Varejistas' ou 'Hotéis/Restaurantes/Cafés' nessa distribuição?
# 
#     Não há, nesta distribuição há diversos pontos invadindo outro segmento.
#     
#     
# - Você consideraria essas classificações como consistentes comparada a sua definição de segmentação de clientes anterior?*
# 
#     Não, anteriormente era possível ver uma separação de segmentos bem definida, porém com esse novo atributo sendo incluído, essa separação se tornou bem mais homogênea.
#  

# > **Nota**: Uma vez que você completou todas as implementações de código e respondeu todas as questões acima com êxito, você pode finalizar seu trabalho exportando um iPython Notebook como um documento HTML. Você pode fazer isso utilizando o menu acima e navegando até  
# **File -> Download as -> HTML (.html)**. Inclua o documento finalizado junto com esse Notebook para o seu envio.
