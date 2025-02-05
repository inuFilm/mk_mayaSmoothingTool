ChatGPT o3-mini highの機能を使用して作成しました
mk_maya_smoothing_tool.pyをシェルフ登録ormayaのscriptEditorで起動

UIが表示されるので
①頂点を選択
②execute

なるべく形状を維持したままsmoothされる（はず）です

![image](https://github.com/user-attachments/assets/ac16605c-d2ad-4312-be09-7f983f8b1198)

パラメーターは基本的にSmooth Amountと　Iterationの2つで大まかに操作してください

Freeze Borderは端っこの頂点を固定します

UIの構成はblenderAddonで既にある下記の影響を受けています（ソフトウェア差で実装出来ない機能が多くあるので、blenderユーザーは下記からご購入してみてください、良いアドオンです）
https://bartoszstyperek.gumroad.com/l/vol_smooth
