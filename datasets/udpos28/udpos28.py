import conllu
import os

import datasets


# _CITATION = """\
# @misc{11234/1-3687,
# title = {Universal Dependencies 2.8.1},
# author = {Zeman, Daniel and Nivre, Joakim and Abrams, Mitchell and Ackermann, Elia and Aepli, No{\"e}mi and Aghaei, Hamid and Agi{\'c}, {\v Z}eljko and Ahmadi, Amir and Ahrenberg, Lars and Ajede,
# Chika Kennedy and Aleksandravi{\v c}i{\=u}t{\.e}, Gabriel{\.e} and Alfina, Ika and Antonsen, Lene and Aplonova, Katya and Aquino, Angelina and Aragon, Carolina and Aranzabe, Maria Jesus and Ar{\i}can, Bilge Nas and Arnard{\'o}ttir, {\t H}{\'o}runn and Arutie, Gashaw and Arwidarasti, Jessica Naraiswari and Asahara, Masayuki and Aslan, Deniz Baran and Ateyah, Luma and Atmaca, Furkan and Attia, Mohammed and Atutxa, Aitziber and Augustinus, Liesbeth and Badmaeva, Elena and Balasubramani, Keerthana and Ballesteros, Miguel and Banerjee, Esha and Bank, Sebastian and Barbu Mititelu, Verginica and Barkarson, Starkaður and Basmov, Victoria and Batchelor, Colin and Bauer, John and Bedir, Seyyit Talha and Bengoetxea, Kepa and Berk, G{\"o}zde and Berzak, Yevgeni and Bhat, Irshad Ahmad and Bhat, Riyaz Ahmad and Biagetti, Erica and Bick, Eckhard and Bielinskien{\.e}, Agn{\.e} and Bjarnad{\'o}ttir, Krist{\'{\i}}n and Blokland, Rogier and Bobicev, Victoria and Boizou, Lo{\"{\i}}c and Borges V{\"o}lker, Emanuel and B{\"o}rstell, Carl and Bosco, Cristina and Bouma, Gosse and Bowman, Sam and Boyd, Adriane and Braggaar, Anouck and Brokait{\.e}, Kristina and Burchardt, Aljoscha and Candito, Marie and Caron, Bernard and Caron, Gauthier and Cassidy, Lauren and Cavalcanti, Tatiana and Cebiro{\u g}lu Eryi{\u g}it, G{\"u}l{\c s}en and Cecchini, Flavio Massimiliano and Celano, Giuseppe G. A. and {\v C}{\'e}pl{\"o}, Slavom{\'{\i}}r and Cesur, Neslihan and Cetin, Savas and {\c C}etino{\u g}lu, {\"O}zlem and Chalub, Fabricio and Chauhan, Shweta and Chi, Ethan and Chika, Taishi and Cho, Yongseok and Choi, Jinho and Chun, Jayeol and Cignarella, Alessandra T. and Cinkov{\'a}, Silvie and Collomb, Aur{\'e}lie and {\c C}{\"o}ltekin, {\c C}a{\u g}r{\i} and Connor, Miriam and Courtin, Marine and Cristescu, Mihaela and Daniel, Philemon. and Davidson, Elizabeth and de Marneffe, Marie-Catherine and de Paiva, Valeria and Derin, Mehmet Oguz and de Souza, Elvis and Diaz de Ilarraza, Arantza and Dickerson, Carly and Dinakaramani, Arawinda and Di Nuovo, Elisa and Dione, Bamba and Dirix, Peter and Dobrovoljc, Kaja and Dozat, Timothy and Droganova, Kira and Dwivedi, Puneet and Eckhoff, Hanne and Eiche, Sandra and Eli, Marhaba and Elkahky, Ali and Ephrem, Binyam and Erina, Olga and Erjavec, Toma{\v z} and Etienne, Aline and Evelyn, Wograine and Facundes, Sidney and Farkas, Rich{\'a}rd and Fernanda, Mar{\'{\i}}lia and Fernandez Alcalde, Hector and Foster, Jennifer and Freitas, Cl{\'a}udia and Fujita, Kazunori and Gajdo{\v s}ov{\'a}, Katar{\'{\i}}na and Galbraith, Daniel and Garcia, Marcos and G{\"a}rdenfors, Moa and Garza, Sebastian and Gerardi, Fabr{\'{\i}}cio Ferraz and Gerdes, Kim and Ginter, Filip and Godoy, Gustavo and Goenaga, Iakes and Gojenola, Koldo and G{\"o}k{\i}rmak, Memduh and Goldberg, Yoav and G{\'o}mez Guinovart, Xavier and Gonz{\'a}lez Saavedra,
# Berta and Grici{\=u}t{\.e}, Bernadeta and Grioni, Matias and Grobol,
# Lo{\"{\i}}c and Gr{\=
# u}z{\={\i}}tis, Normunds and Guillaume, Bruno and Guillot-Barbance, C{\'e}line and G{\"u}ng{\"o}r, Tunga and Habash, Nizar and Hafsteinsson, Hinrik and Haji{\v c}, Jan and Haji{\v c} jr., Jan and H{\"a}m{\"a}l{\"a}inen, Mika and H{\`a} M{\~y}, Linh and Han, Na-Rae and Hanifmuti, Muhammad Yudistira and Hardwick, Sam and Harris, Kim and Haug, Dag and Heinecke, Johannes and Hellwig, Oliver and Hennig, Felix and Hladk{\'a}, Barbora and Hlav{\'a}{\v c}ov{\'a}, Jaroslava and Hociung, Florinel and Hohle, Petter and Huber, Eva and Hwang, Jena and Ikeda, Takumi and Ingason, Anton Karl and Ion, Radu and Irimia, Elena and Ishola, {\d O}l{\'a}j{\'{\i}}d{\'e} and Ito, Kaoru and Jel{\'{\i}}nek, Tom{\'a}{\v s} and Jha, Apoorva and Johannsen, Anders and J{\'o}nsd{\'o}ttir, Hildur and J{\o}rgensen, Fredrik and Juutinen, Markus and K, Sarveswaran and Ka{\c s}{\i}kara, H{\"u}ner and Kaasen, Andre and Kabaeva, Nadezhda and Kahane, Sylvain and Kanayama, Hiroshi and Kanerva, Jenna and Kara, Neslihan and Katz, Boris and Kayadelen, Tolga and Kenney, Jessica and Kettnerov{\'a}, V{\'a}clava and Kirchner, Jesse and Klementieva, Elena and K{\"o}hn, Arne and K{\"o}ksal, Abdullatif and Kopacewicz, Kamil and Korkiakangas, Timo and Kotsyba, Natalia and Kovalevskait{\.e}, Jolanta and Krek, Simon and Krishnamurthy, Parameswari and Kuyruk{\c c}u, O{\u g}uzhan and Kuzgun, Asl{\i} and Kwak, Sookyoung and Laippala, Veronika and Lam, Lucia and Lambertino, Lorenzo and Lando, Tatiana and Larasati, Septina Dian and Lavrentiev, Alexei and Lee, John and L{\^e} H{\`{\^o}}ng, Phương and Lenci, Alessandro and Lertpradit, Saran and Leung, Herman and Levina, Maria and Li, Cheuk Ying and Li, Josie and Li, Keying and Li, Yuan and Lim, {KyungTae} and Lima Padovani, Bruna and Lind{\'e}n, Krister and Ljube{\v s}i{\'c}, Nikola and Loginova, Olga and Luthfi, Andry and Luukko, Mikko and Lyashevskaya, Olga and Lynn, Teresa and Macketanz, Vivien and Makazhanov, Aibek and Mandl, Michael and Manning, Christopher and Manurung, Ruli and Mar{\c s}an, B{\"u}{\c s}ra and M{\u a}r{\u a}nduc, C{\u a}t{\u a}lina and Mare{\v c}ek, David and Marheinecke, Katrin and Mart{\'{\i}}nez Alonso, H{\'e}ctor and Martins, Andr{\'e} and Ma{\v s}ek, Jan and Matsuda, Hiroshi and Matsumoto, Yuji and Mazzei, Alessandro and {McDonald}, Ryan and {McGuinness}, Sarah and Mendon{\c c}a, Gustavo and Miekka, Niko and Mischenkova, Karina and Misirpashayeva, Margarita and Missil{\"a}, Anna and Mititelu, C{\u a}t{\u a}lin and Mitrofan, Maria and Miyao, Yusuke and Mojiri Foroushani, {AmirHossein} and Moln{\'a}r, Judit and Moloodi, Amirsaeid and Montemagni, Simonetta and More, Amir and Moreno Romero, Laura and Moretti, Giovanni and Mori, Keiko Sophie and Mori, Shinsuke and Morioka, Tomohiko and Moro, Shigeki and Mortensen, Bjartur and Moskalevskyi, Bohdan and Muischnek, Kadri and Munro, Robert and Murawaki, Yugo and M{\"u}{\"u}risep, Kaili and Nainwani, Pinkey and Nakhl{\'e}, Mariam and Navarro Hor{\~n}iacek, Juan Ignacio and Nedoluzhko,
# Anna and Ne{\v s}pore-B{\=e}rzkalne, Gunta and Nevaci, Manuela and Nguy{\~{\^e}}n Th{\d i}, Lương and Nguy{\~{\^e}}n Th{\d i} Minh, Huy{\`{\^e}}n and Nikaido, Yoshihiro and Nikolaev, Vitaly and Nitisaroj, Rattima and Nourian, Alireza and Nurmi, Hanna and Ojala, Stina and Ojha, Atul Kr. and Ol{\'u}{\`o}kun, Ad{\'e}day{\d o}̀ and Omura, Mai and Onwuegbuzia, Emeka and Osenova, Petya and {\"O}stling, Robert and {\O}vrelid, Lilja and {\"O}zate{\c s}, {\c S}aziye Bet{\"u}l and {\"O}z{\c c}elik, Merve and {\"O}zg{\"u}r, Arzucan and {\"O}zt{\"u}rk Ba{\c s}aran, Balk{\i}z and Park, Hyunji Hayley and Partanen, Niko and Pascual, Elena and Passarotti, Marco and Patejuk, Agnieszka and Paulino-Passos, Guilherme and Peljak-{\L}api{\'n}ska, Angelika and Peng, Siyao and Perez, Cenel-Augusto and Perkova, Natalia and Perrier, Guy and Petrov, Slav and Petrova, Daria and Phelan, Jason and Piitulainen, Jussi and Pirinen, Tommi A and Pitler, Emily and Plank, Barbara and Poibeau, Thierry and Ponomareva, Larisa and Popel, Martin and Pretkalni{\c n}a, Lauma and Pr{\'e}vost, Sophie and Prokopidis, Prokopis and Przepi{\'o}rkowski, Adam and Puolakainen, Tiina and Pyysalo, Sampo and Qi, Peng and R{\"a}{\"a}bis, Andriela and Rademaker, Alexandre and Rama, Taraka and Ramasamy, Loganathan and Ramisch, Carlos and Rashel, Fam and Rasooli, Mohammad Sadegh and Ravishankar, Vinit and Real, Livy and Rebeja, Petru and Reddy, Siva and Rehm, Georg and Riabov, Ivan and Rie{\ss}ler, Michael and Rimkut{\.e}, Erika and Rinaldi, Larissa and Rituma, Laura and Rocha, Luisa and R{\"o}gnvaldsson, Eir{\'{\i}}kur and Romanenko, Mykhailo and Rosa, Rudolf and Roșca, Valentin and Rovati, Davide and Rudina, Olga and Rueter, Jack and R{\'u}narsson, Kristj{\'a}n and Sadde, Shoval and Safari, Pegah and Sagot, Beno{\^{\i}}t and Sahala, Aleksi and Saleh, Shadi and Salomoni, Alessio and Samard{\v z}i{\'c}, Tanja and Samson, Stephanie and Sanguinetti, Manuela and San{\i}yar, Ezgi and S{\"a}rg,
# Dage and Saul{\={\i}}te, Baiba and Sawanakunanon, Yanin and Saxena, Shefali and Scannell, Kevin and Scarlata, Salvatore and Schneider, Nathan and Schuster, Sebastian and Schwartz, Lane and Seddah, Djam{\'e} and Seeker, Wolfgang and Seraji, Mojgan and Shen, Mo and Shimada, Atsuko and Shirasu, Hiroyuki and Shishkina, Yana and Shohibussirri, Muh and Sichinava, Dmitry and Siewert, Janine and Sigurðsson, Einar Freyr and Silveira, Aline and Silveira, Natalia and Simi, Maria and Simionescu, Radu and Simk{\'o}, Katalin and {\v S}imkov{\'a}, M{\'a}ria and Simov, Kiril and Skachedubova, Maria and Smith, Aaron and Soares-Bastos, Isabela and Spadine, Carolyn and Sprugnoli, Rachele and Steingr{\'{\i}}msson, Stein{\t h}{\'o}r and Stella, Antonio and Straka, Milan and Strickland, Emmett and Strnadov{\'a}, Jana and Suhr, Alane and Sulestio, Yogi Lesmana and Sulubacak, Umut and Suzuki, Shingo and Sz{\'a}nt{\'o}, Zsolt and Taji, Dima and Takahashi, Yuta and Tamburini, Fabio and Tan, Mary Ann C. and Tanaka, Takaaki and Tella, Samson and Tellier, Isabelle and Testori, Marinella and Thomas, Guillaume and Torga, Liisi and Toska, Marsida and Trosterud, Trond and Trukhina, Anna and Tsarfaty, Reut and T{\"u}rk, Utku and Tyers, Francis and Uematsu, Sumire and Untilov, Roman and Ure{\v s}ov{\'a}, Zde{\v n}ka and Uria, Larraitz and Uszkoreit, Hans and Utka, Andrius and Vajjala, Sowmya and van der Goot, Rob and Vanhove, Martine and van Niekerk, Daniel and van Noord, Gertjan and Varga, Viktor and Villemonte de la Clergerie, Eric and Vincze, Veronika and Vlasova, Natalia and Wakasa, Aya and Wallenberg, Joel C. and Wallin, Lars and Walsh, Abigail and Wang, Jing Xian and Washington, Jonathan North and Wendt, Maximilan and Widmer, Paul and Williams, Seyi and Wir{\'e}n, Mats and Wittern, Christian and Woldemariam, Tsegay and Wong, Tak-sum and Wr{\'o}blewska, Alina and Yako, Mary and Yamashita, Kayo and Yamazaki, Naoki and Yan, Chunxiao and Yasuoka, Koichi and Yavrumyan, Marat M. and Yenice, Arife Bet{\"u}l and Y{\i}ld{\i}z, Olcay Taner and Yu, Zhuoran and {\v Z}abokrtsk{\'y}, Zden{\v e}k and Zahra, Shorouq and Zeldes, Amir and Zhu, Hanzhi and Zhuravleva, Anna and Ziane, Rayan},
# url = {http://hdl.handle.net/11234/1-3687},
# note = {{LINDAT}/{CLARIAH}-{CZ} digital library at the Institute of Formal and Applied Linguistics ({{\'U}FAL}), Faculty of Mathematics and Physics, Charles University},
# copyright = {Licence Universal Dependencies v2.8},
# year = {2021} }
# """  # noqa: W605

_DESCRIPTION = """\
Universal Dependencies is a project that seeks to develop cross-linguistically consistent treebank annotation for many languages, with the goal of facilitating multilingual parser development, cross-lingual learning, and parsing research from a language typology perspective. The annotation scheme is based on (universal) Stanford dependencies (de Marneffe et al., 2006, 2008, 2014), Google universal part-of-speech tags (Petrov et al., 2012), and the Interset interlingua for morphosyntactic tagsets (Zeman, 2008).
"""

_URL = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3687/ud-treebanks-v2.8.tgz"

# _UD_DATASETS = {}
# for dirname in os.listdir():
#     if not os.path.isdir(dirname):
#         continue
#     lang, dataset = dirname.replace("UD_", "").split("-")
#     train, dev, test = None, None, None
#     for filename in os.listdir(dirname):
#         if not filename.endswith(".conllu"):
#             continue
#         id_, split = filename.replace(".conllu", "").split("-ud-")
#         langid, dataid = id_.split("_")
#         print(lang, dataset, langid, dataid, split)
#         if langid not in _UD_DATASETS:
#             _UD_DATASETS[langid] = {}
#         if split not in _UD_DATASETS[langid]:
#             _UD_DATASETS[langid][split] = []
#         _UD_DATASETS[langid][split].append(os.path.join(dirname, filename))
_UD_DATASETS = {
    'akk': {
        'test': ['UD_Akkadian-RIAO/akk_riao-ud-test.conllu', 'UD_Akkadian-PISANDUB/akk_pisandub-ud-test.conllu']
    },
    'hy': {
        'train': ['UD_Armenian-ArmTDP/hy_armtdp-ud-train.conllu'],
        'test': ['UD_Armenian-ArmTDP/hy_armtdp-ud-test.conllu'],
        'dev': ['UD_Armenian-ArmTDP/hy_armtdp-ud-dev.conllu']
    },
    'cy': {
        'train': ['UD_Welsh-CCG/cy_ccg-ud-train.conllu'],
        'dev': ['UD_Welsh-CCG/cy_ccg-ud-dev.conllu'],
        'test': ['UD_Welsh-CCG/cy_ccg-ud-test.conllu']
    },
    'no': {
        'test': [
            'UD_Norwegian-Nynorsk/no_nynorsk-ud-test.conllu', 'UD_Norwegian-Bokmaal/no_bokmaal-ud-test.conllu',
            'UD_Norwegian-NynorskLIA/no_nynorsklia-ud-test.conllu'
        ],
        'dev': [
            'UD_Norwegian-Nynorsk/no_nynorsk-ud-dev.conllu', 'UD_Norwegian-Bokmaal/no_bokmaal-ud-dev.conllu',
            'UD_Norwegian-NynorskLIA/no_nynorsklia-ud-dev.conllu'
        ],
        'train': [
            'UD_Norwegian-Nynorsk/no_nynorsk-ud-train.conllu', 'UD_Norwegian-Bokmaal/no_bokmaal-ud-train.conllu',
            'UD_Norwegian-NynorskLIA/no_nynorsklia-ud-train.conllu'
        ]
    },
    'orv': {
        'test': ['UD_Old_East_Slavic-TOROT/orv_torot-ud-test.conllu', 'UD_Old_East_Slavic-RNC/orv_rnc-ud-test.conllu'],
        'train':
        ['UD_Old_East_Slavic-TOROT/orv_torot-ud-train.conllu', 'UD_Old_East_Slavic-RNC/orv_rnc-ud-train.conllu'],
        'dev': ['UD_Old_East_Slavic-TOROT/orv_torot-ud-dev.conllu']
    },
    'en': {
        'train': [
            'UD_English-LinES/en_lines-ud-train.conllu', 'UD_English-EWT/en_ewt-ud-train.conllu',
            'UD_English-GUMReddit/en_gumreddit-ud-train.conllu', 'UD_English-GUM/en_gum-ud-train.conllu',
            'UD_English-ESL/en_esl-ud-train.conllu', 'UD_English-ParTUT/en_partut-ud-train.conllu'
        ],
        'dev': [
            'UD_English-LinES/en_lines-ud-dev.conllu', 'UD_English-EWT/en_ewt-ud-dev.conllu',
            'UD_English-GUMReddit/en_gumreddit-ud-dev.conllu', 'UD_English-GUM/en_gum-ud-dev.conllu',
            'UD_English-ESL/en_esl-ud-dev.conllu', 'UD_English-ParTUT/en_partut-ud-dev.conllu'
        ],
        'test': [
            'UD_English-LinES/en_lines-ud-test.conllu', 'UD_English-PUD/en_pud-ud-test.conllu',
            'UD_English-EWT/en_ewt-ud-test.conllu', 'UD_English-Pronouns/en_pronouns-ud-test.conllu',
            'UD_English-GUMReddit/en_gumreddit-ud-test.conllu', 'UD_English-GUM/en_gum-ud-test.conllu',
            'UD_English-ESL/en_esl-ud-test.conllu', 'UD_English-ParTUT/en_partut-ud-test.conllu'
        ]
    },
    'sq': {
        'test': ['UD_Albanian-TSA/sq_tsa-ud-test.conllu']
    },
    'fr': {
        'dev': [
            'UD_French-Sequoia/fr_sequoia-ud-dev.conllu', 'UD_French-FTB/fr_ftb-ud-dev.conllu',
            'UD_French-ParTUT/fr_partut-ud-dev.conllu', 'UD_French-Spoken/fr_spoken-ud-dev.conllu',
            'UD_French-GSD/fr_gsd-ud-dev.conllu'
        ],
        'train': [
            'UD_French-Sequoia/fr_sequoia-ud-train.conllu', 'UD_French-FTB/fr_ftb-ud-train.conllu',
            'UD_French-ParTUT/fr_partut-ud-train.conllu', 'UD_French-Spoken/fr_spoken-ud-train.conllu',
            'UD_French-GSD/fr_gsd-ud-train.conllu'
        ],
        'test': [
            'UD_French-Sequoia/fr_sequoia-ud-test.conllu', 'UD_French-FTB/fr_ftb-ud-test.conllu',
            'UD_French-ParTUT/fr_partut-ud-test.conllu', 'UD_French-PUD/fr_pud-ud-test.conllu',
            'UD_French-Spoken/fr_spoken-ud-test.conllu', 'UD_French-GSD/fr_gsd-ud-test.conllu',
            'UD_French-FQB/fr_fqb-ud-test.conllu'
        ]
    },
    'qhe': {
        'dev': ['UD_Hindi_English-HIENCS/qhe_hiencs-ud-dev.conllu'],
        'test': ['UD_Hindi_English-HIENCS/qhe_hiencs-ud-test.conllu'],
        'train': ['UD_Hindi_English-HIENCS/qhe_hiencs-ud-train.conllu']
    },
    'sl': {
        'test': ['UD_Slovenian-SST/sl_sst-ud-test.conllu', 'UD_Slovenian-SSJ/sl_ssj-ud-test.conllu'],
        'train': ['UD_Slovenian-SST/sl_sst-ud-train.conllu', 'UD_Slovenian-SSJ/sl_ssj-ud-train.conllu'],
        'dev': ['UD_Slovenian-SSJ/sl_ssj-ud-dev.conllu']
    },
    'gub': {
        'test': ['UD_Guajajara-TuDeT/gub_tudet-ud-test.conllu']
    },
    'kmr': {
        'test': ['UD_Kurmanji-MG/kmr_mg-ud-test.conllu'],
        'train': ['UD_Kurmanji-MG/kmr_mg-ud-train.conllu']
    },
    'it': {
        'test': [
            'UD_Italian-PUD/it_pud-ud-test.conllu', 'UD_Italian-PoSTWITA/it_postwita-ud-test.conllu',
            'UD_Italian-ISDT/it_isdt-ud-test.conllu', 'UD_Italian-VIT/it_vit-ud-test.conllu',
            'UD_Italian-ParTUT/it_partut-ud-test.conllu', 'UD_Italian-Valico/it_valico-ud-test.conllu',
            'UD_Italian-TWITTIRO/it_twittiro-ud-test.conllu'
        ],
        'dev': [
            'UD_Italian-PoSTWITA/it_postwita-ud-dev.conllu', 'UD_Italian-ISDT/it_isdt-ud-dev.conllu',
            'UD_Italian-VIT/it_vit-ud-dev.conllu', 'UD_Italian-ParTUT/it_partut-ud-dev.conllu',
            'UD_Italian-TWITTIRO/it_twittiro-ud-dev.conllu'
        ],
        'train': [
            'UD_Italian-PoSTWITA/it_postwita-ud-train.conllu', 'UD_Italian-ISDT/it_isdt-ud-train.conllu',
            'UD_Italian-VIT/it_vit-ud-train.conllu', 'UD_Italian-ParTUT/it_partut-ud-train.conllu',
            'UD_Italian-TWITTIRO/it_twittiro-ud-train.conllu'
        ]
    },
    'tr': {
        'test': [
            'UD_Turkish-GB/tr_gb-ud-test.conllu', 'UD_Turkish-Kenet/tr_kenet-ud-test.conllu',
            'UD_Turkish-FrameNet/tr_framenet-ud-test.conllu', 'UD_Turkish-IMST/tr_imst-ud-test.conllu',
            'UD_Turkish-Penn/tr_penn-ud-test.conllu', 'UD_Turkish-PUD/tr_pud-ud-test.conllu',
            'UD_Turkish-Tourism/tr_tourism-ud-test.conllu', 'UD_Turkish-BOUN/tr_boun-ud-test.conllu'
        ],
        'dev': [
            'UD_Turkish-Kenet/tr_kenet-ud-dev.conllu', 'UD_Turkish-FrameNet/tr_framenet-ud-dev.conllu',
            'UD_Turkish-IMST/tr_imst-ud-dev.conllu', 'UD_Turkish-Penn/tr_penn-ud-dev.conllu',
            'UD_Turkish-Tourism/tr_tourism-ud-dev.conllu', 'UD_Turkish-BOUN/tr_boun-ud-dev.conllu'
        ],
        'train': [
            'UD_Turkish-Kenet/tr_kenet-ud-train.conllu', 'UD_Turkish-FrameNet/tr_framenet-ud-train.conllu',
            'UD_Turkish-IMST/tr_imst-ud-train.conllu', 'UD_Turkish-Penn/tr_penn-ud-train.conllu',
            'UD_Turkish-Tourism/tr_tourism-ud-train.conllu', 'UD_Turkish-BOUN/tr_boun-ud-train.conllu'
        ]
    },
    'fi': {
        'test': [
            'UD_Finnish-FTB/fi_ftb-ud-test.conllu', 'UD_Finnish-OOD/fi_ood-ud-test.conllu',
            'UD_Finnish-TDT/fi_tdt-ud-test.conllu', 'UD_Finnish-PUD/fi_pud-ud-test.conllu'
        ],
        'dev': ['UD_Finnish-FTB/fi_ftb-ud-dev.conllu', 'UD_Finnish-TDT/fi_tdt-ud-dev.conllu'],
        'train': ['UD_Finnish-FTB/fi_ftb-ud-train.conllu', 'UD_Finnish-TDT/fi_tdt-ud-train.conllu']
    },
    'id': {
        'test': [
            'UD_Indonesian-GSD/id_gsd-ud-test.conllu', 'UD_Indonesian-CSUI/id_csui-ud-test.conllu',
            'UD_Indonesian-PUD/id_pud-ud-test.conllu'
        ],
        'train': ['UD_Indonesian-GSD/id_gsd-ud-train.conllu', 'UD_Indonesian-CSUI/id_csui-ud-train.conllu'],
        'dev': ['UD_Indonesian-GSD/id_gsd-ud-dev.conllu']
    },
    'uk': {
        'test': ['UD_Ukrainian-IU/uk_iu-ud-test.conllu'],
        'dev': ['UD_Ukrainian-IU/uk_iu-ud-dev.conllu'],
        'train': ['UD_Ukrainian-IU/uk_iu-ud-train.conllu']
    },
    'nl': {
        'dev': ['UD_Dutch-LassySmall/nl_lassysmall-ud-dev.conllu', 'UD_Dutch-Alpino/nl_alpino-ud-dev.conllu'],
        'train': ['UD_Dutch-LassySmall/nl_lassysmall-ud-train.conllu', 'UD_Dutch-Alpino/nl_alpino-ud-train.conllu'],
        'test': ['UD_Dutch-LassySmall/nl_lassysmall-ud-test.conllu', 'UD_Dutch-Alpino/nl_alpino-ud-test.conllu']
    },
    'pl': {
        'test': [
            'UD_Polish-PDB/pl_pdb-ud-test.conllu', 'UD_Polish-PUD/pl_pud-ud-test.conllu',
            'UD_Polish-LFG/pl_lfg-ud-test.conllu'
        ],
        'train': ['UD_Polish-PDB/pl_pdb-ud-train.conllu', 'UD_Polish-LFG/pl_lfg-ud-train.conllu'],
        'dev': ['UD_Polish-PDB/pl_pdb-ud-dev.conllu', 'UD_Polish-LFG/pl_lfg-ud-dev.conllu']
    },
    'pt': {
        'test': [
            'UD_Portuguese-Bosque/pt_bosque-ud-test.conllu', 'UD_Portuguese-PUD/pt_pud-ud-test.conllu',
            'UD_Portuguese-GSD/pt_gsd-ud-test.conllu'
        ],
        'dev': ['UD_Portuguese-Bosque/pt_bosque-ud-dev.conllu', 'UD_Portuguese-GSD/pt_gsd-ud-dev.conllu'],
        'train': ['UD_Portuguese-Bosque/pt_bosque-ud-train.conllu', 'UD_Portuguese-GSD/pt_gsd-ud-train.conllu']
    },
    'kk': {
        'test': ['UD_Kazakh-KTB/kk_ktb-ud-test.conllu'],
        'train': ['UD_Kazakh-KTB/kk_ktb-ud-train.conllu']
    },
    'la': {
        'test': [
            'UD_Latin-ITTB/la_ittb-ud-test.conllu', 'UD_Latin-Perseus/la_perseus-ud-test.conllu',
            'UD_Latin-LLCT/la_llct-ud-test.conllu', 'UD_Latin-PROIEL/la_proiel-ud-test.conllu',
            'UD_Latin-UDante/la_udante-ud-test.conllu'
        ],
        'dev': [
            'UD_Latin-ITTB/la_ittb-ud-dev.conllu', 'UD_Latin-LLCT/la_llct-ud-dev.conllu',
            'UD_Latin-PROIEL/la_proiel-ud-dev.conllu', 'UD_Latin-UDante/la_udante-ud-dev.conllu'
        ],
        'train': [
            'UD_Latin-ITTB/la_ittb-ud-train.conllu', 'UD_Latin-Perseus/la_perseus-ud-train.conllu',
            'UD_Latin-LLCT/la_llct-ud-train.conllu', 'UD_Latin-PROIEL/la_proiel-ud-train.conllu',
            'UD_Latin-UDante/la_udante-ud-train.conllu'
        ]
    },
    'fro': {
        'train': ['UD_Old_French-SRCMF/fro_srcmf-ud-train.conllu'],
        'test': ['UD_Old_French-SRCMF/fro_srcmf-ud-test.conllu'],
        'dev': ['UD_Old_French-SRCMF/fro_srcmf-ud-dev.conllu']
    },
    'es': {
        'test': [
            'UD_Spanish-PUD/es_pud-ud-test.conllu', 'UD_Spanish-GSD/es_gsd-ud-test.conllu',
            'UD_Spanish-AnCora/es_ancora-ud-test.conllu'
        ],
        'train': ['UD_Spanish-GSD/es_gsd-ud-train.conllu', 'UD_Spanish-AnCora/es_ancora-ud-train.conllu'],
        'dev': ['UD_Spanish-GSD/es_gsd-ud-dev.conllu', 'UD_Spanish-AnCora/es_ancora-ud-dev.conllu']
    },
    'bxr': {
        'test': ['UD_Buryat-BDT/bxr_bdt-ud-test.conllu'],
        'train': ['UD_Buryat-BDT/bxr_bdt-ud-train.conllu']
    },
    'urb': {
        'test': ['UD_Kaapor-TuDeT/urb_tudet-ud-test.conllu']
    },
    'ko': {
        'test': [
            'UD_Korean-PUD/ko_pud-ud-test.conllu', 'UD_Korean-GSD/ko_gsd-ud-test.conllu',
            'UD_Korean-Kaist/ko_kaist-ud-test.conllu'
        ],
        'train': ['UD_Korean-GSD/ko_gsd-ud-train.conllu', 'UD_Korean-Kaist/ko_kaist-ud-train.conllu'],
        'dev': ['UD_Korean-GSD/ko_gsd-ud-dev.conllu', 'UD_Korean-Kaist/ko_kaist-ud-dev.conllu']
    },
    'et': {
        'train': ['UD_Estonian-EDT/et_edt-ud-train.conllu', 'UD_Estonian-EWT/et_ewt-ud-train.conllu'],
        'test': ['UD_Estonian-EDT/et_edt-ud-test.conllu', 'UD_Estonian-EWT/et_ewt-ud-test.conllu'],
        'dev': ['UD_Estonian-EDT/et_edt-ud-dev.conllu', 'UD_Estonian-EWT/et_ewt-ud-dev.conllu']
    },
    'hr': {
        'train': ['UD_Croatian-SET/hr_set-ud-train.conllu'],
        'dev': ['UD_Croatian-SET/hr_set-ud-dev.conllu'],
        'test': ['UD_Croatian-SET/hr_set-ud-test.conllu']
    },
    'got': {
        'test': ['UD_Gothic-PROIEL/got_proiel-ud-test.conllu'],
        'dev': ['UD_Gothic-PROIEL/got_proiel-ud-dev.conllu'],
        'train': ['UD_Gothic-PROIEL/got_proiel-ud-train.conllu']
    },
    'swl': {
        'test': ['UD_Swedish_Sign_Language-SSLC/swl_sslc-ud-test.conllu'],
        'train': ['UD_Swedish_Sign_Language-SSLC/swl_sslc-ud-train.conllu'],
        'dev': ['UD_Swedish_Sign_Language-SSLC/swl_sslc-ud-dev.conllu']
    },
    'gsw': {
        'test': ['UD_Swiss_German-UZH/gsw_uzh-ud-test.conllu']
    },
    'aii': {
        'test': ['UD_Assyrian-AS/aii_as-ud-test.conllu']
    },
    'sme': {
        'train': ['UD_North_Sami-Giella/sme_giella-ud-train.conllu'],
        'test': ['UD_North_Sami-Giella/sme_giella-ud-test.conllu']
    },
    'pcm': {
        'test': ['UD_Naija-NSC/pcm_nsc-ud-test.conllu'],
        'dev': ['UD_Naija-NSC/pcm_nsc-ud-dev.conllu'],
        'train': ['UD_Naija-NSC/pcm_nsc-ud-train.conllu']
    },
    'de': {
        'test': [
            'UD_German-LIT/de_lit-ud-test.conllu', 'UD_German-HDT/de_hdt-ud-test.conllu',
            'UD_German-GSD/de_gsd-ud-test.conllu', 'UD_German-PUD/de_pud-ud-test.conllu'
        ],
        'train': ['UD_German-HDT/de_hdt-ud-train.conllu', 'UD_German-GSD/de_gsd-ud-train.conllu'],
        'dev': ['UD_German-HDT/de_hdt-ud-dev.conllu', 'UD_German-GSD/de_gsd-ud-dev.conllu']
    },
    'lv': {
        'train': ['UD_Latvian-LVTB/lv_lvtb-ud-train.conllu'],
        'dev': ['UD_Latvian-LVTB/lv_lvtb-ud-dev.conllu'],
        'test': ['UD_Latvian-LVTB/lv_lvtb-ud-test.conllu']
    },
    'zh': {
        'train': ['UD_Chinese-GSDSimp/zh_gsdsimp-ud-train.conllu', 'UD_Chinese-GSD/zh_gsd-ud-train.conllu'],
        'test': [
            'UD_Chinese-GSDSimp/zh_gsdsimp-ud-test.conllu', 'UD_Chinese-HK/zh_hk-ud-test.conllu',
            'UD_Chinese-CFL/zh_cfl-ud-test.conllu', 'UD_Chinese-PUD/zh_pud-ud-test.conllu',
            'UD_Chinese-GSD/zh_gsd-ud-test.conllu'
        ],
        'dev': ['UD_Chinese-GSDSimp/zh_gsdsimp-ud-dev.conllu', 'UD_Chinese-GSD/zh_gsd-ud-dev.conllu']
    },
    'tl': {
        'test': ['UD_Tagalog-Ugnayan/tl_ugnayan-ud-test.conllu', 'UD_Tagalog-TRG/tl_trg-ud-test.conllu']
    },
    'bm': {
        'test': ['UD_Bambara-CRB/bm_crb-ud-test.conllu']
    },
    'lt': {
        'dev': ['UD_Lithuanian-ALKSNIS/lt_alksnis-ud-dev.conllu', 'UD_Lithuanian-HSE/lt_hse-ud-dev.conllu'],
        'test': ['UD_Lithuanian-ALKSNIS/lt_alksnis-ud-test.conllu', 'UD_Lithuanian-HSE/lt_hse-ud-test.conllu'],
        'train': ['UD_Lithuanian-ALKSNIS/lt_alksnis-ud-train.conllu', 'UD_Lithuanian-HSE/lt_hse-ud-train.conllu']
    },
    'gl': {
        'test': ['UD_Galician-CTG/gl_ctg-ud-test.conllu', 'UD_Galician-TreeGal/gl_treegal-ud-test.conllu'],
        'dev': ['UD_Galician-CTG/gl_ctg-ud-dev.conllu'],
        'train': ['UD_Galician-CTG/gl_ctg-ud-train.conllu', 'UD_Galician-TreeGal/gl_treegal-ud-train.conllu']
    },
    'vi': {
        'test': ['UD_Vietnamese-VTB/vi_vtb-ud-test.conllu'],
        'dev': ['UD_Vietnamese-VTB/vi_vtb-ud-dev.conllu'],
        'train': ['UD_Vietnamese-VTB/vi_vtb-ud-train.conllu']
    },
    'am': {
        'test': ['UD_Amharic-ATT/am_att-ud-test.conllu']
    },
    'el': {
        'train': ['UD_Greek-GDT/el_gdt-ud-train.conllu'],
        'test': ['UD_Greek-GDT/el_gdt-ud-test.conllu'],
        'dev': ['UD_Greek-GDT/el_gdt-ud-dev.conllu']
    },
    'ca': {
        'test': ['UD_Catalan-AnCora/ca_ancora-ud-test.conllu'],
        'dev': ['UD_Catalan-AnCora/ca_ancora-ud-dev.conllu'],
        'train': ['UD_Catalan-AnCora/ca_ancora-ud-train.conllu']
    },
    'soj': {
        'test': ['UD_Soi-AHA/soj_aha-ud-test.conllu']
    },
    'sv': {
        'dev': ['UD_Swedish-LinES/sv_lines-ud-dev.conllu', 'UD_Swedish-Talbanken/sv_talbanken-ud-dev.conllu'],
        'test': [
            'UD_Swedish-LinES/sv_lines-ud-test.conllu', 'UD_Swedish-Talbanken/sv_talbanken-ud-test.conllu',
            'UD_Swedish-PUD/sv_pud-ud-test.conllu'
        ],
        'train': ['UD_Swedish-LinES/sv_lines-ud-train.conllu', 'UD_Swedish-Talbanken/sv_talbanken-ud-train.conllu']
    },
    'ess': {
        'test': ['UD_Yupik-SLI/ess_sli-ud-test.conllu']
    },
    'ru': {
        'dev': [
            'UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu', 'UD_Russian-Taiga/ru_taiga-ud-dev.conllu',
            'UD_Russian-GSD/ru_gsd-ud-dev.conllu'
        ],
        'test': [
            'UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu', 'UD_Russian-Taiga/ru_taiga-ud-test.conllu',
            'UD_Russian-PUD/ru_pud-ud-test.conllu', 'UD_Russian-GSD/ru_gsd-ud-test.conllu'
        ],
        'train': [
            'UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu', 'UD_Russian-Taiga/ru_taiga-ud-train.conllu',
            'UD_Russian-GSD/ru_gsd-ud-train.conllu'
        ]
    },
    'cs': {
        'dev': [
            'UD_Czech-PDT/cs_pdt-ud-dev.conllu', 'UD_Czech-FicTree/cs_fictree-ud-dev.conllu',
            'UD_Czech-CAC/cs_cac-ud-dev.conllu', 'UD_Czech-CLTT/cs_cltt-ud-dev.conllu'
        ],
        'train': [
            'UD_Czech-PDT/cs_pdt-ud-train.conllu', 'UD_Czech-FicTree/cs_fictree-ud-train.conllu',
            'UD_Czech-CAC/cs_cac-ud-train.conllu', 'UD_Czech-CLTT/cs_cltt-ud-train.conllu'
        ],
        'test': [
            'UD_Czech-PDT/cs_pdt-ud-test.conllu', 'UD_Czech-FicTree/cs_fictree-ud-test.conllu',
            'UD_Czech-CAC/cs_cac-ud-test.conllu', 'UD_Czech-CLTT/cs_cltt-ud-test.conllu',
            'UD_Czech-PUD/cs_pud-ud-test.conllu'
        ]
    },
    'bej': {
        'test': ['UD_Beja-NSC/bej_nsc-ud-test.conllu']
    },
    'myv': {
        'test': ['UD_Erzya-JR/myv_jr-ud-test.conllu']
    },
    'bho': {
        'test': ['UD_Bhojpuri-BHTB/bho_bhtb-ud-test.conllu']
    },
    'th': {
        'test': ['UD_Thai-PUD/th_pud-ud-test.conllu']
    },
    'mr': {
        'train': ['UD_Marathi-UFAL/mr_ufal-ud-train.conllu'],
        'dev': ['UD_Marathi-UFAL/mr_ufal-ud-dev.conllu'],
        'test': ['UD_Marathi-UFAL/mr_ufal-ud-test.conllu']
    },
    'eu': {
        'test': ['UD_Basque-BDT/eu_bdt-ud-test.conllu'],
        'dev': ['UD_Basque-BDT/eu_bdt-ud-dev.conllu'],
        'train': ['UD_Basque-BDT/eu_bdt-ud-train.conllu']
    },
    'sk': {
        'dev': ['UD_Slovak-SNK/sk_snk-ud-dev.conllu'],
        'train': ['UD_Slovak-SNK/sk_snk-ud-train.conllu'],
        'test': ['UD_Slovak-SNK/sk_snk-ud-test.conllu']
    },
    'quc': {
        'test': ['UD_Kiche-IU/quc_iu-ud-test.conllu']
    },
    'yo': {
        'test': ['UD_Yoruba-YTB/yo_ytb-ud-test.conllu']
    },
    'wbp': {
        'test': ['UD_Warlpiri-UFAL/wbp_ufal-ud-test.conllu']
    },
    'nds': {
        'test': ['UD_Low_Saxon-LSDC/nds_lsdc-ud-test.conllu']
    },
    'ta': {
        'train': ['UD_Tamil-TTB/ta_ttb-ud-train.conllu'],
        'dev': ['UD_Tamil-TTB/ta_ttb-ud-dev.conllu'],
        'test': ['UD_Tamil-TTB/ta_ttb-ud-test.conllu', 'UD_Tamil-MWTT/ta_mwtt-ud-test.conllu']
    },
    'mt': {
        'test': ['UD_Maltese-MUDT/mt_mudt-ud-test.conllu'],
        'train': ['UD_Maltese-MUDT/mt_mudt-ud-train.conllu'],
        'dev': ['UD_Maltese-MUDT/mt_mudt-ud-dev.conllu']
    },
    'grc': {
        'dev':
        ['UD_Ancient_Greek-Perseus/grc_perseus-ud-dev.conllu', 'UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu'],
        'test':
        ['UD_Ancient_Greek-Perseus/grc_perseus-ud-test.conllu', 'UD_Ancient_Greek-PROIEL/grc_proiel-ud-test.conllu'],
        'train':
        ['UD_Ancient_Greek-Perseus/grc_perseus-ud-train.conllu', 'UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu']
    },
    'is': {
        'train': ['UD_Icelandic-IcePaHC/is_icepahc-ud-train.conllu', 'UD_Icelandic-Modern/is_modern-ud-train.conllu'],
        'test': [
            'UD_Icelandic-IcePaHC/is_icepahc-ud-test.conllu', 'UD_Icelandic-Modern/is_modern-ud-test.conllu',
            'UD_Icelandic-PUD/is_pud-ud-test.conllu'
        ],
        'dev': ['UD_Icelandic-IcePaHC/is_icepahc-ud-dev.conllu', 'UD_Icelandic-Modern/is_modern-ud-dev.conllu']
    },
    'gun': {
        'test':
        ['UD_Mbya_Guarani-Thomas/gun_thomas-ud-test.conllu', 'UD_Mbya_Guarani-Dooley/gun_dooley-ud-test.conllu']
    },
    'ur': {
        'train': ['UD_Urdu-UDTB/ur_udtb-ud-train.conllu'],
        'dev': ['UD_Urdu-UDTB/ur_udtb-ud-dev.conllu'],
        'test': ['UD_Urdu-UDTB/ur_udtb-ud-test.conllu']
    },
    'ro': {
        'dev': [
            'UD_Romanian-RRT/ro_rrt-ud-dev.conllu', 'UD_Romanian-Nonstandard/ro_nonstandard-ud-dev.conllu',
            'UD_Romanian-SiMoNERo/ro_simonero-ud-dev.conllu'
        ],
        'test': [
            'UD_Romanian-RRT/ro_rrt-ud-test.conllu', 'UD_Romanian-ArT/ro_art-ud-test.conllu',
            'UD_Romanian-Nonstandard/ro_nonstandard-ud-test.conllu', 'UD_Romanian-SiMoNERo/ro_simonero-ud-test.conllu'
        ],
        'train': [
            'UD_Romanian-RRT/ro_rrt-ud-train.conllu', 'UD_Romanian-Nonstandard/ro_nonstandard-ud-train.conllu',
            'UD_Romanian-SiMoNERo/ro_simonero-ud-train.conllu'
        ]
    },
    'fa': {
        'test': ['UD_Persian-PerDT/fa_perdt-ud-test.conllu', 'UD_Persian-Seraji/fa_seraji-ud-test.conllu'],
        'train': ['UD_Persian-PerDT/fa_perdt-ud-train.conllu', 'UD_Persian-Seraji/fa_seraji-ud-train.conllu'],
        'dev': ['UD_Persian-PerDT/fa_perdt-ud-dev.conllu', 'UD_Persian-Seraji/fa_seraji-ud-dev.conllu']
    },
    'apu': {
        'test': ['UD_Apurina-UFPA/apu_ufpa-ud-test.conllu']
    },
    'ja': {
        'test': [
            'UD_Japanese-Modern/ja_modern-ud-test.conllu', 'UD_Japanese-BCCWJ/ja_bccwj-ud-test.conllu',
            'UD_Japanese-GSD/ja_gsd-ud-test.conllu', 'UD_Japanese-PUD/ja_pud-ud-test.conllu'
        ],
        'train': ['UD_Japanese-BCCWJ/ja_bccwj-ud-train.conllu', 'UD_Japanese-GSD/ja_gsd-ud-train.conllu'],
        'dev': ['UD_Japanese-BCCWJ/ja_bccwj-ud-dev.conllu', 'UD_Japanese-GSD/ja_gsd-ud-dev.conllu']
    },
    'hu': {
        'train': ['UD_Hungarian-Szeged/hu_szeged-ud-train.conllu'],
        'test': ['UD_Hungarian-Szeged/hu_szeged-ud-test.conllu'],
        'dev': ['UD_Hungarian-Szeged/hu_szeged-ud-dev.conllu']
    },
    'hi': {
        'test': ['UD_Hindi-HDTB/hi_hdtb-ud-test.conllu', 'UD_Hindi-PUD/hi_pud-ud-test.conllu'],
        'train': ['UD_Hindi-HDTB/hi_hdtb-ud-train.conllu'],
        'dev': ['UD_Hindi-HDTB/hi_hdtb-ud-dev.conllu']
    },
    'lzh': {
        'dev': ['UD_Classical_Chinese-Kyoto/lzh_kyoto-ud-dev.conllu'],
        'train': ['UD_Classical_Chinese-Kyoto/lzh_kyoto-ud-train.conllu'],
        'test': ['UD_Classical_Chinese-Kyoto/lzh_kyoto-ud-test.conllu']
    },
    'koi': {
        'test': ['UD_Komi_Permyak-UH/koi_uh-ud-test.conllu']
    },
    'fo': {
        'dev': ['UD_Faroese-FarPaHC/fo_farpahc-ud-dev.conllu'],
        'test': ['UD_Faroese-FarPaHC/fo_farpahc-ud-test.conllu', 'UD_Faroese-OFT/fo_oft-ud-test.conllu'],
        'train': ['UD_Faroese-FarPaHC/fo_farpahc-ud-train.conllu']
    },
    'sa': {
        'test': ['UD_Sanskrit-Vedic/sa_vedic-ud-test.conllu', 'UD_Sanskrit-UFAL/sa_ufal-ud-test.conllu'],
        'train': ['UD_Sanskrit-Vedic/sa_vedic-ud-train.conllu']
    },
    'olo': {
        'test': ['UD_Livvi-KKPP/olo_kkpp-ud-test.conllu'],
        'train': ['UD_Livvi-KKPP/olo_kkpp-ud-train.conllu']
    },
    'ar': {
        'train': ['UD_Arabic-NYUAD/ar_nyuad-ud-train.conllu', 'UD_Arabic-PADT/ar_padt-ud-train.conllu'],
        'dev': ['UD_Arabic-NYUAD/ar_nyuad-ud-dev.conllu', 'UD_Arabic-PADT/ar_padt-ud-dev.conllu'],
        'test': [
            'UD_Arabic-NYUAD/ar_nyuad-ud-test.conllu', 'UD_Arabic-PUD/ar_pud-ud-test.conllu',
            'UD_Arabic-PADT/ar_padt-ud-test.conllu'
        ]
    },
    'wo': {
        'test': ['UD_Wolof-WTB/wo_wtb-ud-test.conllu'],
        'dev': ['UD_Wolof-WTB/wo_wtb-ud-dev.conllu'],
        'train': ['UD_Wolof-WTB/wo_wtb-ud-train.conllu']
    },
    'bg': {
        'test': ['UD_Bulgarian-BTB/bg_btb-ud-test.conllu'],
        'train': ['UD_Bulgarian-BTB/bg_btb-ud-train.conllu'],
        'dev': ['UD_Bulgarian-BTB/bg_btb-ud-dev.conllu']
    },
    'aqz': {
        'test': ['UD_Akuntsu-TuDeT/aqz_tudet-ud-test.conllu']
    },
    'mpu': {
        'test': ['UD_Makurap-TuDeT/mpu_tudet-ud-test.conllu']
    },
    'xnr': {
        'test': ['UD_Kangri-KDTB/xnr_kdtb-ud-test.conllu']
    },
    'br': {
        'test': ['UD_Breton-KEB/br_keb-ud-test.conllu']
    },
    'te': {
        'train': ['UD_Telugu-MTG/te_mtg-ud-train.conllu'],
        'test': ['UD_Telugu-MTG/te_mtg-ud-test.conllu'],
        'dev': ['UD_Telugu-MTG/te_mtg-ud-dev.conllu']
    },
    'yue': {
        'test': ['UD_Cantonese-HK/yue_hk-ud-test.conllu']
    },
    'qtd': {
        'test': ['UD_Turkish_German-SAGT/qtd_sagt-ud-test.conllu'],
        'dev': ['UD_Turkish_German-SAGT/qtd_sagt-ud-dev.conllu'],
        'train': ['UD_Turkish_German-SAGT/qtd_sagt-ud-train.conllu']
    },
    'cu': {
        'dev': ['UD_Old_Church_Slavonic-PROIEL/cu_proiel-ud-dev.conllu'],
        'test': ['UD_Old_Church_Slavonic-PROIEL/cu_proiel-ud-test.conllu'],
        'train': ['UD_Old_Church_Slavonic-PROIEL/cu_proiel-ud-train.conllu']
    },
    'krl': {
        'test': ['UD_Karelian-KKPP/krl_kkpp-ud-test.conllu']
    },
    'hsb': {
        'train': ['UD_Upper_Sorbian-UFAL/hsb_ufal-ud-train.conllu'],
        'test': ['UD_Upper_Sorbian-UFAL/hsb_ufal-ud-test.conllu']
    },
    'da': {
        'train': ['UD_Danish-DDT/da_ddt-ud-train.conllu'],
        'dev': ['UD_Danish-DDT/da_ddt-ud-dev.conllu'],
        'test': ['UD_Danish-DDT/da_ddt-ud-test.conllu']
    },
    'ajp': {
        'test': ['UD_South_Levantine_Arabic-MADAR/ajp_madar-ud-test.conllu']
    },
    'kpv': {
        'test': ['UD_Komi_Zyrian-Lattice/kpv_lattice-ud-test.conllu', 'UD_Komi_Zyrian-IKDP/kpv_ikdp-ud-test.conllu']
    },
    'ga': {
        'test': ['UD_Irish-IDT/ga_idt-ud-test.conllu', 'UD_Irish-TwittIrish/ga_twittirish-ud-test.conllu'],
        'train': ['UD_Irish-IDT/ga_idt-ud-train.conllu'],
        'dev': ['UD_Irish-IDT/ga_idt-ud-dev.conllu']
    },
    'nyq': {
        'test': ['UD_Nayini-AHA/nyq_aha-ud-test.conllu']
    },
    'qfn': {
        'test': ['UD_Frisian_Dutch-Fame/qfn_fame-ud-test.conllu']
    },
    'myu': {
        'test': ['UD_Munduruku-TuDeT/myu_tudet-ud-test.conllu']
    },
    'gv': {
        'test': ['UD_Manx-Cadhan/gv_cadhan-ud-test.conllu']
    },
    'sms': {
        'test': ['UD_Skolt_Sami-Giellagas/sms_giellagas-ud-test.conllu']
    },
    'af': {
        'dev': ['UD_Afrikaans-AfriBooms/af_afribooms-ud-dev.conllu'],
        'test': ['UD_Afrikaans-AfriBooms/af_afribooms-ud-test.conllu'],
        'train': ['UD_Afrikaans-AfriBooms/af_afribooms-ud-train.conllu']
    },
    'otk': {
        'test': ['UD_Old_Turkish-Tonqq/otk_tonqq-ud-test.conllu']
    },
    'tpn': {
        'test': ['UD_Tupinamba-TuDeT/tpn_tudet-ud-test.conllu']
    },
    'be': {
        'train': ['UD_Belarusian-HSE/be_hse-ud-train.conllu'],
        'dev': ['UD_Belarusian-HSE/be_hse-ud-dev.conllu'],
        'test': ['UD_Belarusian-HSE/be_hse-ud-test.conllu']
    },
    'cop': {
        'train': ['UD_Coptic-Scriptorium/cop_scriptorium-ud-train.conllu'],
        'dev': ['UD_Coptic-Scriptorium/cop_scriptorium-ud-dev.conllu'],
        'test': ['UD_Coptic-Scriptorium/cop_scriptorium-ud-test.conllu']
    },
    'sr': {
        'train': ['UD_Serbian-SET/sr_set-ud-train.conllu'],
        'dev': ['UD_Serbian-SET/sr_set-ud-dev.conllu'],
        'test': ['UD_Serbian-SET/sr_set-ud-test.conllu']
    },
    'mdf': {
        'test': ['UD_Moksha-JR/mdf_jr-ud-test.conllu']
    },
    'hyw': {
        'test': ['UD_Western_Armenian-ArmTDP/hyw_armtdp-ud-test.conllu'],
        'dev': ['UD_Western_Armenian-ArmTDP/hyw_armtdp-ud-dev.conllu'],
        'train': ['UD_Western_Armenian-ArmTDP/hyw_armtdp-ud-train.conllu']
    },
    'gd': {
        'dev': ['UD_Scottish_Gaelic-ARCOSG/gd_arcosg-ud-dev.conllu'],
        'train': ['UD_Scottish_Gaelic-ARCOSG/gd_arcosg-ud-train.conllu'],
        'test': ['UD_Scottish_Gaelic-ARCOSG/gd_arcosg-ud-test.conllu']
    },
    'kfm': {
        'test': ['UD_Khunsari-AHA/kfm_aha-ud-test.conllu']
    },
    'he': {
        'dev': ['UD_Hebrew-HTB/he_htb-ud-dev.conllu'],
        'train': ['UD_Hebrew-HTB/he_htb-ud-train.conllu'],
        'test': ['UD_Hebrew-HTB/he_htb-ud-test.conllu']
    },
    'ug': {
        'test': ['UD_Uyghur-UDT/ug_udt-ud-test.conllu'],
        'train': ['UD_Uyghur-UDT/ug_udt-ud-train.conllu'],
        'dev': ['UD_Uyghur-UDT/ug_udt-ud-dev.conllu']
    },
    'ckt': {
        'test': ['UD_Chukchi-HSE/ckt_hse-ud-test.conllu']
    }
}


class Udpos28Config(datasets.BuilderConfig):
    """BuilderConfig for Universal dependencies"""
    def __init__(self, **kwargs):
        super(Udpos28Config, self).__init__(**kwargs)

        self.data_url = _URL


class Udpos28(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("2.8.1")  # type: ignore
    # BUILDER_CONFIGS = [Udpos28Config(
    #     name=name,
    #     data_url=_URL,
    # ) for name in _UD_DATASETS]
    BUILDER_CONFIG_CLASS = Udpos28Config

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "idx":
                datasets.Value("string"),
                # "text":
                # datasets.Value("string"),
                "tokens":
                datasets.Sequence(datasets.Value("string")),
                # "lemmas":
                # datasets.Sequence(datasets.Value("string")),
                "labels":
                datasets.Sequence(
                    datasets.features.ClassLabel(names=[
                        "ADJ",
                        "ADP",
                        "ADV",
                        "AUX",
                        "CCONJ",
                        "DET",
                        "INTJ",
                        "NOUN",
                        "NUM",
                        "PART",
                        "PRON",
                        "PROPN",
                        "PUNCT",
                        "SCONJ",
                        "SYM",
                        "VERB",
                        "X",
                    ])),
                # "xpos":
                # datasets.Sequence(datasets.Value("string")),
                # "feats":
                # datasets.Sequence(datasets.Value("string")),
                # "head":
                # datasets.Sequence(datasets.Value("string")),
                # "deprel":
                # datasets.Sequence(datasets.Value("string")),
                # "deps":
                # datasets.Sequence(datasets.Value("string")),
                # "misc":
                # datasets.Sequence(datasets.Value("string")),
            }),
            supervised_keys=None,
            homepage="https://universaldependencies.org/",
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(self.config.data_url)
        data_dir = os.path.join(downloaded_files, "ud-treebanks-v2.8")
        splits = []

        if "-" in self.config.name:
            data_paths = {}
            for name in self.config.name.split("-"):
                if "train" in _UD_DATASETS[name]:
                    data_paths["train"] = [*data_paths.get("train", []), *_UD_DATASETS[name]["train"]]
                if "dev" in _UD_DATASETS[name]:
                    data_paths["dev"] = [*data_paths.get("dev", []), *_UD_DATASETS[name]["dev"]]
                if "test" in _UD_DATASETS[name]:
                    data_paths["test"] = [*data_paths.get("test", []), *_UD_DATASETS[name]["test"]]

        else:
            data_paths = _UD_DATASETS[self.config.name]

        if "train" in data_paths:
            splits.append(
                datasets.SplitGenerator(name=str(datasets.Split.TRAIN),
                                        gen_kwargs={"data_dir": data_dir, "filepaths": data_paths["train"]}))

        if "dev" in data_paths:
            splits.append(
                datasets.SplitGenerator(name=str(datasets.Split.VALIDATION),
                                        gen_kwargs={"data_dir": data_dir, "filepaths": data_paths["dev"]}))

        if "test" in data_paths:
            splits.append(
                datasets.SplitGenerator(name=str(datasets.Split.TEST),
                                        gen_kwargs={"data_dir": data_dir, "filepaths": data_paths["test"]}))

        return splits

    def _generate_examples(self, data_dir, filepaths):
        id = 0
        for path in filepaths:
            path = os.path.join(data_dir, path)
            with open(path, "r", encoding="utf-8") as data_file:
                tokenlist = list(conllu.parse_incr(data_file))
                for sent in tokenlist:
                    if "sent_id" in sent.metadata:
                        idx = sent.metadata["sent_id"]
                    else:
                        idx = id

                    tokens = [token["form"] for token in sent]
                    upos = [token["upos"] for token in sent]
                    if "_" in tokens or "_" in upos:
                        continue

                    # if "text" in sent.metadata:
                    #     txt = sent.metadata["text"]
                    # else:
                    #     txt = " ".join(tokens)

                    yield id, {
                        "idx": str(idx),
                        # "text": txt,
                        "tokens": tokens,
                        # "lemmas": [token["lemma"] for token in sent],
                        "labels": upos,
                        # "xpos": [token["xpos"] for token in sent],
                        # "feats": [str(token["feats"]) for token in sent],
                        # "head": [str(token["head"]) for token in sent],
                        # "deprel": [str(token["deprel"]) for token in sent],
                        # "deps": [str(token["deps"]) for token in sent],
                        # "misc": [str(token["misc"]) for token in sent],
                    }
                    id += 1
