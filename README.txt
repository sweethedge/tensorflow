# tensorflow

■ dimensions 늘리는 법

tf.expand_dims(a, axis = 0)

■ dimensions 줄이는 법

tf.squeeze(d)

■ 니가 만들고 싶은 길이로 랜덤하게 텐서 만드는 법

tf.random.uniform(shape=(2, 3), minval=0, maxval=1)
tf.random.normal(shape=(2,3)) # mean=0, stddev=1
tf.random.truncated_normal(shape=(2, 3)) # mean=0, stddev=1

-- 그걸 Variable로 만들 수도 있음
tf.Variable(tf.random.truncated_normal(shape=(2, 3)))

■ 텐서 0부터 만드는 법
tf.reshape(tf.range(12), shape = (3, 4))

■ tf.range() == np.arange()
a = tf.reshape(tf.random.shuffle(tf.range(12)), shape = (3, 4))

■ a부터 b까지 잘게 쪼개고 싶으면 tf.linspace(start, end, num = n)

■ 미분하고 싶으면
- 일단 constant를 만들고
x = tf.constant(2.0)
y = tf.constant(3.0)
-- 테이프에 넣어서
with tf.GradientTape(persistent=True) as tape:
-- watch에 넣고
	tape.watch(x)
-- tape.gradient(계산식, x)로 미분
dx, dy = tape.gradient(z, [x, y])

0407
■ Gradient Descent
1. 배열을 준비
2. 가중치와 noise도 준비
step = lr * fprime(x)
3. 테이프에 f(x)를 넣고 미분한 값을 assign_sub
x.assign_sub(step)
■ tf.math.multiply() == tf.reduce_prod()

■ model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
□ metrics
니가 혼돈행렬에서 봤던 그거다. precision이나 recall, Fil-score같은 걸 써 볼 수 있겠다.
- accuracy
-- 맞는 걸 맞고, 틀린 걸 틀렸다고 한 비율
- precision
-- 맞는 걸 맞고, 틀린 걸 맞다고 한 것 중 맞는 걸 맞다고 한 비율
- recall
-- 맞는 걸 맞고, 맞는 걸 틀렸다고 한 것 중 맞는 걸 맞다고 한 비율.

precision과 recall은 trade-off 관계에 있다고 얘기를 한다.
보통은 accuracy를 쓰게 되는데, precision이 중요한 경우가 있다고들 한다. 예를 들어서 spam filter를 짜고 싶으면 metrics에 accuracy보다는 precision을 쓰셔야 될 거다.

□ loss 함수
loss 함수를 만든다. model이 함수를 그려주면 input과의 차이를 loss 함수를 만들어서 계산하고, 이에 따라서 loss가 최소가 되도록 조율을 한다.
- categorical_crossentropy
-- mnist처럼 categorical한 데이터를 model이 분류해야 하면 categorical_crossentropy
- mse or rmse
-- 어떤 수치를 예측하고 싶으면 metrics에 mse or rmse, mae
-- binary 문제를 쓸 때는 sigmoid
-- 여러 개 중에 고르고 싶을 때는 softmax
-- 0이 mean이 되고 -1 ~ 1이 되는 활성화함수가 필요하면 tanh(Hyperbolic tangent function)

■ keras.dataset 분석하는 법
1. (x_train, y_train), (x_test, y_test) = 어쩌구.load_data()
2. model = tf.keras.Sequential()
3. model.add(tf.keras.layers.Dense(units=10, input_dim=x_train.shape[1])) (model.add는 하고 싶은 만큼 하시면 된다)
3-1. model.summary()
4. opt = tf.keras.optimizers.RMSprop() # default learning_rate = 0.001
5. model.compile(optimizer=opt, loss='mse', metrics=['mae'])
6. ret = model.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=0)
7. model의 loss 값을 알고 싶으면 tf.keras.Sequential().evaluate(X, Y)
train_loss = model.evaluate(x_train, y_train, verbose=2)
test_loss = model.evaluate(x_test, y_test, verbose=2)
8. 만든 model로 예측하고 싶으면 model.predict(검정 데이터)
y_pred = model.predict(x_test)

9.
plt.ylabel("니가 documentation에서 읽은 y_train이 나타내는 그거")
plt.plot(y_pred, "r-", label="y_pred")
plt.plot(y_test, "b-", label="y_test")
# loc : int or string or pair of floats, default: ‘upper right’
plt.legend(loc='best')
plt.show()
