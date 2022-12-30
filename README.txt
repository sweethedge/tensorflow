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