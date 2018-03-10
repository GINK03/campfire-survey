# Campfire Survey
Campfireはクラウドファウンディングで何らかを企画して、お金を募って実行するというものです。  

クラウドファウンディングに諸々と気になっており、でも勝算のないまま突っ込むのは無謀だとも思い簡単にサーベイしたいと思います  

## 1. Campfireのカテゴリごとの出資額の分布

Campfireはアニメや地域創生などで投資対象になることがありますが、カテゴリにより必要な予算感というのはだいぶ異なってきます。  

カテゴリが何らかの分布を持つとき、バイオリン図というもので簡単に表現することができます。  

Plotlyというサービスを利用することで、Pandasのデータフレームを引数に簡単に描画することができます。

<div align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/4949982/37239330-f54fe4e6-247c-11e8-9f6a-6a2ecda9747f.png">
</div>

金額をLogのオーダを取っているので差がわかりにくいですが、10 ~ 100倍の予算の差があることがわかります  

## 2. Campfireのカテゴリごとの出資額の達成率の分布

達成率という視点で見ていきますと、{実際集まった額}/{目標額}という視点でこれをlogをとってカテゴリごとにバイオリン図を描くとこのようになります。  
<div align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/4949982/37239380-e8b3984e-247d-11e8-8dd5-f0a467056f59.png">
</div>
達成率で行くとそこそこ達成するか、達成率が極めて悪いかの二極化していることがわかりました。  

アニメや地方創生は安定して好成績を達成しますが、ビジネスやガジェットは失敗率が高そうです。

## 3. Twitterのフォロワー数と出資額の関係

これも直感的には関係がありそうですよね、ということで仮設をまず立てて見ていきます。  

<div align="center">
  <img width="500px" src="https://user-images.githubusercontent.com/4949982/37239278-0b6b929e-247c-11e8-8058-e429aa7b25e3.png">
</div>
横軸を募集を募る人のフォロワー数として、縦軸を実際に出資された金額を見ると、相関は最小二乗法でとりあえず＋で、分布の偏りがあることわかり、フォロワーがある一定以上多いと、観測頻度も上がっているように見えて、単純に、このクラウドファウンディングを利用するのに、ソーシャルの影響力を一定以上超えている人が多いとかもありそうです。  

## 4. Campfireの目的金額の達成されるケースと達成されないケースの違いは何なのでしょうか
なんだか今回多くの
