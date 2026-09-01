Ezek az ábráink eddig. Mindegyik fájlnévhez csatolok egy kisebb leírást ide.


ci_analysis\-0.867_-0.867_1.733\results\ -->

    eigenvalues_improved.png: az egyes állapotokhoz tartozó sajátértékek.

    eigenvector_components_improved.png: az egyes állapotokhoz tartozó sajátvektorok komponenseinek a valós része. Az előjel-váltások miatt szerintünk anno jó volt ezeket kiábrázolni.

    gamma_evolution_improved.png: az egyes topológiai fázisok akkumulációja.

    gamma_heatmap_improved.png: ezt is meghagytam a kódokban, mivel szerintem ha más paraméterezésekkel, vagy esetleg nxn-es kiterjesztéskor, hasznos lehet.

    R_thetas_projections_improved.png: ezt még ki kell találnom, hogyan tudom szebben ábrázolni. Egyelőre ez a paraméter-útvonalunk projekciója minden síkra, beleértve az ortogonális síkot is a triviális CI-seamre. Olyat is tudnék csinálni, hopgy 3D-n kiábrázolom az 1,1,1 egyenest, és a projekciókat kiábrázolom a xy, yz, és xz síkokra. Valahogy így: https://matplotlib.org/stable/gallery/mplot3d/contourf3d_2.html

    tau_abs_23_improved.png, tau_evolution_improved.png és tau_abs_32_improved.png és: A nemadiabatikus csatolási tagok komplex abszolútértékei és imaginárius értékei kiábrázolva a szögek függvényében.

    Vx_components_improved.png, Va_components_improved.png és Vx_Va_comparison.png: a Va és Vx potrenciálok ábrái, illetve különbségeik (azok 100-al felskálázva) is kiábrázolva a szögparaméter függvényében.


ci_analysis\3d_and_CIs\ -->

    ci_points_orthogonal_plane.png: Az összes CI, beleértve a triviális eseté (projekció) is, az 1,1,1 egyenesre merőleges síkon kiábrázolva. Hozzá a d_CI sugarú kör is be lett rajzolva, hogy illeszkedjen a jegyzethez.

    ci_seam_3d_no_inset.png és ci_seam_3d.png: ezek hasonlóak az előbbi ábrával, csak itt most 3D-n lett minden kiábrázolva. Az egyik helyen bele is nagyítottam az origóba, ami most a triviális CI-ba lett elhelyezve.


ci_analysis\together\ -->

    combined_gamma_23_comparison.png és combined_gamma_32_comparison.png: ezek a topológia fázsok akkumulációit ábrázolják, a theta szög függvényében

    combined_tau_23_comparison.png és combined_tau_32_comparison.png: ezek a nemadiabatikus csatolási tagok akkumulációit ábrázolják ki, a theta szög függvényében