xslt_expan =  ('''<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:tei="http://www.tei-c.org/ns/1.0">
                <xsl:output method="text" indent="no"/>
                    <!-- nur Kindelemente von 'body' auswerten -->
                    <xsl:template match="tei:TEI">
                    <xsl:apply-templates select="tei:text/tei:body/*"/>
                    </xsl:template>
                    <!-- Kopfzeile nicht übernehmen -->
                    <xsl:template match="tei:fw"/>
                    <!-- Nur expan übernehmen -->
                    <xsl:template match="tei:abbr"/>
                    <!-- Ausschluss Editorial question -->
                    <xsl:template match="tei:note[@type='editorial-question']">
                    </xsl:template>
                    <!-- Ausschluss Editorial comment -->
                    <xsl:template match="tei:note[@type='editorial-comment']">
                    </xsl:template>
                    <xsl:template match="tei:note[@type='inscription']">
                    <xsl:apply-templates/>
                    <xsl:text>§</xsl:text>
                    </xsl:template>
                    <!-- Ausschluss label-->
                    <xsl:template match="tei:label"/>
                    <!-- Ausschluss seg -->
                    <xsl:template match="tei:seg"/>
                    <!-- Ausschluss Interpunktion-->
                    <xsl:template match="tei:pc"/>
                    <!-- Modellierung von Zusätzen -->
                    <!-- linebeginnings als einfacher Zeilenumbruch -->
                    <xsl:template match="tei:lb">
                        <xsl:text>§</xsl:text>
                    </xsl:template>
                    <!-- lb break ="no" ohne Zeilenumbruch -->
                    <xsl:template match="tei:lb[@break='no']">
                        <xsl:text>+</xsl:text>
                    </xsl:template>
                    <!-- cb ohne Zeilenumbruch -->
                    <xsl:template match="tei:cb"/>
                    <!-- Von oxygen eingefügte Leerzeichenketten in choice entfernen -->
                    <xsl:template match="tei:choice/text()"/>
                    <!-- Von oxygen eingefügte Leerzeichenketten in div entfernen -->
                    <xsl:template match="tei:div/text()"/>
                    <!-- Von oxygen eingefügte Leerzeichenketten in Liste entfernen -->
                    <xsl:template match="tei:list/text()"/>
                    <!-- Von oxygen eingefügte Leerzeichenketten in Liste entfernen -->
                    <xsl:template match="tei:subst/text()"/>
                </xsl:stylesheet>''')

xslt_abbr =  ('''<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:tei="http://www.tei-c.org/ns/1.0">
                <xsl:output method="text" indent="no"/>
                    <!-- nur Kindelemente von 'body' auswerten -->
                    <xsl:template match="tei:TEI">
                    <xsl:apply-templates select="tei:text/tei:body/*"/>
                    </xsl:template>
                    <!-- Kopfzeile nicht übernehmen -->
                    <xsl:template match="tei:fw"/>
                    <!-- Nur expan übernehmen -->
                    <xsl:template match="tei:expan"/>
                    <!-- Ausschluss Editorial question -->
                    <xsl:template match="tei:note[@type='editorial-question']">
                    </xsl:template>
                    <!-- Ausschluss Editorial comment -->
                    <xsl:template match="tei:note[@type='editorial-comment']">
                    </xsl:template>
                    <xsl:template match="tei:note[@type='inscription']">
                    <xsl:apply-templates/>
                    <xsl:text>§</xsl:text>
                    </xsl:template>
                    <!-- Ausschluss label-->
                    <xsl:template match="tei:label"/>
                    <!-- Ausschluss seg -->
                    <xsl:template match="tei:seg"/>
                    <!-- Ausschluss Interpunktion-->
                    <xsl:template match="tei:pc"/>
                    <!-- Modellierung von Zusätzen -->
                    <!-- linebeginnings als einfacher Zeilenumbruch -->
                    <xsl:template match="tei:lb">
                        <xsl:text>§</xsl:text>
                    </xsl:template>
                    <!-- lb break ="no" ohne Zeilenumbruch -->
                    <xsl:template match="tei:lb[@break='no']">
                        <xsl:text>+</xsl:text>
                    </xsl:template>
                    <!-- cb ohne Zeilenumbruch -->
                    <xsl:template match="tei:cb"/>
                    <!-- Von oxygen eingefügte Leerzeichenketten in choice entfernen -->
                    <xsl:template match="tei:choice/text()"/>
                    <!-- Von oxygen eingefügte Leerzeichenketten in div entfernen -->
                    <xsl:template match="tei:div/text()"/>
                    <!-- Von oxygen eingefügte Leerzeichenketten in Liste entfernen -->
                    <xsl:template match="tei:list/text()"/>
                    <!-- Von oxygen eingefügte Leerzeichenketten in Liste entfernen -->
                    <xsl:template match="tei:subst/text()"/>
                </xsl:stylesheet>''')